#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Dataset Split Script

This script processes multiple dataset directories and merges them into a single
training/validation split with sequential frame numbering across all datasets.

Features:
- Process multiple dataset folders automatically
- Sequential frame numbering across all datasets
- Split merged dataset into train/validation sets with configurable ratio
- Copy cam, lidar, annotation, and groundtruth files
- Support for using existing frame_mapping.txt and indices files
- Reproducible splits with random seed

Usage:
    # Process all datasets in /data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/ with 80% training data
    python split_dataset_all.py --source_root /data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared --train_ratio 0.8
    
    # Use existing frame mapping and indices files
    python split_dataset_all.py --use_existing_mapping --mapping_file frame_mapping.txt --output_dir ./dataset3/
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import random
from collections import defaultdict


class MultiDatasetSplitter:
    def __init__(self, 
                 source_root="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared",
                 output_dir="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/",
                 train_ratio=0.8,
                 random_seed=42,
                 use_existing_mapping=False,
                 mapping_file=None):
        """
        Initialize the multi-dataset splitter
        
        Args:
            source_root (str): Root directory containing multiple dataset folders
            output_dir (str): Output directory for split dataset
            train_ratio (float): Ratio of training data (0.0 to 1.0)
            random_seed (int): Random seed for reproducible splits
            use_existing_mapping (bool): Whether to use existing frame mapping
            mapping_file (str): Path to existing frame_mapping.txt file
        """
        self.source_root = Path(source_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_existing_mapping = use_existing_mapping
        self.mapping_file = mapping_file
        
        # Output directories
        self.train_cam_dir = self.output_dir / "train" / "cam"
        self.train_lidar_dir = self.output_dir / "train" / "lidar"
        self.train_annotation_dir = self.output_dir / "train" / "annotations"
        self.train_gt_meta_dir = self.output_dir / "train" / "gt_meta"
        self.train_gt_under_ego_dir = self.output_dir / "train" / "gt_under_ego"
        self.train_gt_under_world_dir = self.output_dir / "train" / "gt_under_world"
        
        self.val_cam_dir = self.output_dir / "val" / "cam"
        self.val_lidar_dir = self.output_dir / "val" / "lidar"
        self.val_annotation_dir = self.output_dir / "val" / "annotations"
        self.val_gt_meta_dir = self.output_dir / "val" / "gt_meta"
        self.val_gt_under_ego_dir = self.output_dir / "val" / "gt_under_ego"
        self.val_gt_under_world_dir = self.output_dir / "val" / "gt_under_world"
        
        # Set random seed
        random.seed(self.random_seed)
        
        # Store frame mapping information
        self.frame_mapping = []  # List of (dataset_folder, original_idx, new_idx)
    
    def create_output_directories(self):
        """Create all necessary output directories"""
        directories = [
            self.train_cam_dir,
            self.train_lidar_dir, 
            self.train_annotation_dir,
            self.train_gt_meta_dir,
            self.train_gt_under_ego_dir,
            self.train_gt_under_world_dir,
            self.val_cam_dir,
            self.val_lidar_dir,
            self.val_annotation_dir,
            self.val_gt_meta_dir,
            self.val_gt_under_ego_dir,
            self.val_gt_under_world_dir
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def find_dataset_folders(self):
        """Find all dataset folders in the source root"""
        dataset_folders = []
        
        for item in self.source_root.iterdir():
            if item.is_dir() and (item / "CoastGuard1" / "cam").exists():
                dataset_folders.append(item)
        
        # Sort folders to ensure consistent ordering
        dataset_folders.sort()
        print(f"Found {len(dataset_folders)} dataset folders:")
        for folder in dataset_folders:
            print(f"  - {folder.name}")
        
        return dataset_folders
    
    def get_frame_indices_from_folder(self, dataset_folder):
        """Get all frame indices from a specific dataset folder"""
        lidar_dir = dataset_folder / "CoastGuard1" / "lidar"
        ppi_files = glob.glob(str(lidar_dir / "*_lidars_ppi.png"))
        
        indices = []
        for file_path in ppi_files:
            filename = os.path.basename(file_path)
            # Extract index from filename like "0000_lidars_ppi.png"
            index = filename.split("_")[0]
            indices.append(index)
        
        indices.sort()
        return indices
    
    def read_frame_mapping(self):
        """Read frame mapping from existing frame_mapping.txt file"""
        if self.mapping_file:
            mapping_path = Path(self.mapping_file)
        else:
            mapping_path = self.output_dir / "frame_mapping.txt"
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Frame mapping file not found: {mapping_path}")
        
        print(f"Reading frame mapping from: {mapping_path}")
        
        self.frame_mapping = []
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    dataset_folder = parts[0]
                    original_idx = parts[1]
                    new_idx = parts[2]
                    
                    # Convert dataset folder name to Path object
                    dataset_path = self.source_root / dataset_folder
                    self.frame_mapping.append((dataset_path, original_idx, new_idx))
        
        print(f"Loaded {len(self.frame_mapping)} frame mappings")
        return True
    
    def read_existing_indices(self):
        """Read train and validation indices from existing files"""
        train_file = self.output_dir / "train_indices.txt"
        val_file = self.output_dir / "val_indices.txt"
        
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(
                f"Index files not found. Expected:\n"
                f"  {train_file}\n"
                f"  {val_file}"
            )
        
        # Read train indices
        with open(train_file, 'r') as f:
            train_indices = [line.strip() for line in f if line.strip()]
        
        # Read val indices  
        with open(val_file, 'r') as f:
            val_indices = [line.strip() for line in f if line.strip()]
        
        print(f"Read {len(train_indices)} training indices from {train_file}")
        print(f"Read {len(val_indices)} validation indices from {val_file}")
        
        return train_indices, val_indices
    
    def build_frame_mapping(self, dataset_folders):
        """Build mapping from original indices to new sequential indices"""
        print("\nBuilding frame mapping...")
        
        current_new_idx = 0
        
        for dataset_folder in dataset_folders:
            original_indices = self.get_frame_indices_from_folder(dataset_folder)
            
            print(f"Dataset {dataset_folder.name}: {len(original_indices)} frames")
            
            for original_idx in original_indices:
                new_idx = f"{current_new_idx:04d}"
                self.frame_mapping.append((dataset_folder, original_idx, new_idx))
                current_new_idx += 1
        
        print(f"Total frames across all datasets: {len(self.frame_mapping)}")
        
        # Save frame mapping for reference
        mapping_file = self.output_dir / "frame_mapping.txt"
        with open(mapping_file, 'w') as f:
            f.write("# Dataset_Folder Original_Index New_Index\n")
            for dataset_folder, original_idx, new_idx in self.frame_mapping:
                f.write(f"{dataset_folder.name} {original_idx} {new_idx}\n")
        
        print(f"Frame mapping saved to: {mapping_file}")
    
    def split_indices(self):
        """Split the new sequential indices into train and validation sets"""
        total_samples = len(self.frame_mapping)
        train_count = int(total_samples * self.train_ratio)
        
        # Create list of new indices
        all_new_indices = [new_idx for _, _, new_idx in self.frame_mapping]
        
        # Shuffle indices randomly
        shuffled_indices = all_new_indices.copy()
        random.shuffle(shuffled_indices)
        
        train_indices = shuffled_indices[:train_count]
        val_indices = shuffled_indices[train_count:]
        
        print(f"\nTotal samples: {total_samples}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        
        return sorted(train_indices), sorted(val_indices)
    
    def copy_file_with_new_name(self, src_file, dest_dir, old_idx, new_idx):
        """Copy a file and rename it with the new index"""
        if not src_file.exists():
            return False
        
        src_filename = src_file.name
        # Replace the old index with new index in filename
        new_filename = src_filename.replace(old_idx, new_idx, 1)
        dest_path = dest_dir / new_filename
        
        shutil.copy2(src_file, dest_path)
        return True
    
    def copy_files_for_indices(self, target_indices, split_name):
        """Copy files for given indices to the appropriate split directories"""
        print(f"\nCopying {split_name} files...")
        
        # Determine output directories based on split
        if split_name == "training":
            cam_dir = self.train_cam_dir
            lidar_dir = self.train_lidar_dir
            annotation_dir = self.train_annotation_dir
            gt_meta_dir = self.train_gt_meta_dir
            gt_under_ego_dir = self.train_gt_under_ego_dir
            gt_under_world_dir = self.train_gt_under_world_dir
        else:  # validation
            cam_dir = self.val_cam_dir
            lidar_dir = self.val_lidar_dir
            annotation_dir = self.val_annotation_dir
            gt_meta_dir = self.val_gt_meta_dir
            gt_under_ego_dir = self.val_gt_under_ego_dir
            gt_under_world_dir = self.val_gt_under_world_dir
        
        # Convert target indices to set for faster lookup
        target_set = set(target_indices)
        
        copied_counts = defaultdict(int)
        
        for dataset_folder, original_idx, new_idx in self.frame_mapping:
            if new_idx not in target_set:
                continue
            
            # Copy camera files
            cam_source_dir = dataset_folder / "CoastGuard1" / "cam"
            cam_patterns = [
                f"{original_idx}_cam_*.png"
            ]
            
            for pattern in cam_patterns:
                cam_files = glob.glob(str(cam_source_dir / pattern))
                for cam_file in cam_files:
                    if self.copy_file_with_new_name(Path(cam_file), cam_dir, original_idx, new_idx):
                        copied_counts['cam'] += 1
            
            # Copy lidar files
            lidar_source_dir = dataset_folder / "CoastGuard1" / "lidar"
            lidar_patterns = [
                f"{original_idx}_lidars_lidar.pcd",
                f"{original_idx}_lidars_ppi.png"
            ]
            
            for pattern in lidar_patterns:
                lidar_files = glob.glob(str(lidar_source_dir / pattern))
                for lidar_file in lidar_files:
                    if self.copy_file_with_new_name(Path(lidar_file), lidar_dir, original_idx, new_idx):
                        copied_counts['lidar'] += 1
            
            # Copy annotation files
            ann_source_dir = dataset_folder / "groundtruth" / "lidar_ppi"
            if ann_source_dir.exists():
                ann_patterns = [
                    f"{original_idx}_ppi_2dbox.yaml",
                    f"{original_idx}_pcd_3dbox.yaml"
                ]
                
                for pattern in ann_patterns:
                    ann_files = glob.glob(str(ann_source_dir / pattern))
                    for ann_file in ann_files:
                        if self.copy_file_with_new_name(Path(ann_file), annotation_dir, original_idx, new_idx):
                            copied_counts['annotation'] += 1
            
            # Copy gt_meta files
            gt_meta_source_dir = dataset_folder / "groundtruth" / "gt_meta"
            if gt_meta_source_dir.exists():
                gt_meta_patterns = [f"{original_idx}*"]
                for pattern in gt_meta_patterns:
                    gt_files = glob.glob(str(gt_meta_source_dir / pattern))
                    for gt_file in gt_files:
                        if self.copy_file_with_new_name(Path(gt_file), gt_meta_dir, original_idx, new_idx):
                            copied_counts['gt_meta'] += 1
            
            # Copy gt_under_ego files
            gt_ego_source_dir = dataset_folder / "groundtruth" / "gt_under_ego"
            if gt_ego_source_dir.exists():
                gt_ego_patterns = [f"{original_idx}*"]
                for pattern in gt_ego_patterns:
                    gt_files = glob.glob(str(gt_ego_source_dir / pattern))
                    for gt_file in gt_files:
                        if self.copy_file_with_new_name(Path(gt_file), gt_under_ego_dir, original_idx, new_idx):
                            copied_counts['gt_under_ego'] += 1
            
            # Copy gt_under_world files
            gt_world_source_dir = dataset_folder / "groundtruth" / "gt_under_world"
            if gt_world_source_dir.exists():
                gt_world_patterns = [f"{original_idx}*"]
                for pattern in gt_world_patterns:
                    gt_files = glob.glob(str(gt_world_source_dir / pattern))
                    for gt_file in gt_files:
                        if self.copy_file_with_new_name(Path(gt_file), gt_under_world_dir, original_idx, new_idx):
                            copied_counts['gt_under_world'] += 1
        
        # Print copy statistics
        for file_type, count in copied_counts.items():
            print(f"  Copied {count} {file_type} files")
    
    def create_file_lists(self, train_indices, val_indices):
        """Create text files listing the indices for each split"""
        # Save train indices
        train_list_file = self.output_dir / "train_indices.txt"
        with open(train_list_file, 'w') as f:
            for idx in sorted(train_indices):
                f.write(f"{idx}\n")
        
        # Save val indices
        val_list_file = self.output_dir / "val_indices.txt"
        with open(val_list_file, 'w') as f:
            for idx in sorted(val_indices):
                f.write(f"{idx}\n")
        
        print(f"\nSaved train indices to: {train_list_file}")
        print(f"Saved validation indices to: {val_list_file}")
    
    def process_datasets(self):
        """Main function to process multiple datasets"""
        print("Starting multi-dataset processing...")
        print(f"Source root: {self.source_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Use existing mapping: {self.use_existing_mapping}")
        if not self.use_existing_mapping:
            print(f"Train ratio: {self.train_ratio}")
        
        # Check if source root exists
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source root directory not found: {self.source_root}")
        
        # Create output directories
        self.create_output_directories()
        
        if self.use_existing_mapping:
            # Use existing frame mapping and indices
            print("\nUsing existing frame mapping and indices...")
            self.read_frame_mapping()
            train_indices, val_indices = self.read_existing_indices()
        else:
            # Find all dataset folders and build new mapping
            dataset_folders = self.find_dataset_folders()
            if not dataset_folders:
                raise ValueError(f"No dataset folders found in {self.source_root}")
            
            # Build frame mapping with sequential numbering
            self.build_frame_mapping(dataset_folders)
            
            # Split indices into train and validation
            train_indices, val_indices = self.split_indices()
            
            # Create file lists
            self.create_file_lists(train_indices, val_indices)
        
        # Copy files for training set
        self.copy_files_for_indices(train_indices, "training")
        
        # Copy files for validation set
        self.copy_files_for_indices(val_indices, "validation")
        
        print(f"\n=== Multi-Dataset Processing Complete ===")
        if not self.use_existing_mapping:
            print(f"Processed {len(self.find_dataset_folders())} dataset folders")
        print(f"Total frames: {len(self.frame_mapping)}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Output saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process multiple datasets and split into train/validation sets")
    parser.add_argument(
        "--source_root",
        type=str,
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared",
        help="Root directory containing multiple dataset folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset3/",
        help="Output directory for split dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training data (0.0 to 1.0), default: 0.9"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits, default: 42"
    )
    parser.add_argument(
        "--use_existing_mapping",
        action="store_true",
        help="Use existing frame_mapping.txt and indices files instead of creating new ones"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        help="Path to existing frame_mapping.txt file (optional, defaults to output_dir/frame_mapping.txt)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_existing_mapping and not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0.0 and 1.0")
    
    # Initialize splitter and run
    splitter = MultiDatasetSplitter(
        source_root=args.source_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_existing_mapping=args.use_existing_mapping,
        mapping_file=args.mapping_file
    )
    
    try:
        splitter.process_datasets()
    except Exception as e:
        print(f"Error during multi-dataset processing: {e}")
        raise


if __name__ == "__main__":
    main()