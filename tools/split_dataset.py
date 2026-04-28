#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Split Script

This script splits the dataset into training and validation sets.
It creates directories for cam, lidar images, annotations, and additional 
groundtruth directories (gt_meta, gt_under_ego, gt_under_world).

Features:
- Split dataset into train/validation sets with configurable ratio
- Copy cam, lidar, annotation, and groundtruth files
- Support for using existing train_indices.txt and val_indices.txt files
- Reproducible splits with random seed

Usage:
    # Generate new split with 80% training data
    python split_dataset.py --train_ratio 0.8
    
    # Use existing indices files to split additional data
    python split_dataset.py --use_existing_indices
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import random


class DatasetSplitter:
    def __init__(self, 
                 base_dir="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/2025-10-18_22-26-46",
                 output_dir="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/",
                 train_ratio=0.8,
                 random_seed=42,
                 use_existing_indices=False):
        """
        Initialize the dataset splitter
        
        Args:
            base_dir (str): Base directory containing the original dataset
            output_dir (str): Output directory for split dataset
            train_ratio (float): Ratio of training data (0.0 to 1.0)
            random_seed (int): Random seed for reproducible splits
            use_existing_indices (bool): Whether to use existing indices files instead of generating new ones
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_existing_indices = use_existing_indices
        
        # Set paths
        self.cam_dir = self.base_dir / "CoastGuard1" / "cam"
        self.lidar_dir = self.base_dir / "CoastGuard1" / "lidar"
        self.annotation_dir = self.base_dir / "groundtruth" / "lidar_ppi"
        self.gt_meta_dir = self.base_dir / "groundtruth" / "gt_meta"
        self.gt_under_ego_dir = self.base_dir / "groundtruth" / "gt_under_ego"
        self.gt_under_world_dir = self.base_dir / "groundtruth" / "gt_under_world"
        
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
    
    def get_sample_indices(self):
        """Get all available sample indices from lidar PPI files"""
        ppi_files = glob.glob(str(self.lidar_dir / "*_lidars_ppi.png"))
        indices = []
        
        for file_path in ppi_files:
            filename = os.path.basename(file_path)
            # Extract index from filename like "0000_lidars_ppi.png"
            index = filename.split("_")[0]
            indices.append(index)
        
        indices.sort()
        return indices
    
    def read_indices_from_files(self):
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
    
    def split_indices(self, indices):
        """Split indices into train and validation sets"""
        total_samples = len(indices)
        train_count = int(total_samples * self.train_ratio)
        
        # Shuffle indices randomly
        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)
        
        train_indices = shuffled_indices[:train_count]
        val_indices = shuffled_indices[train_count:]
        
        print(f"Total samples: {total_samples}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        
        return sorted(train_indices), sorted(val_indices)
    
    def copy_cam_files(self, indices, output_dir, split_name):
        """Copy camera files for given indices"""
        print(f"\nCopying {split_name} camera files...")
        copied_count = 0
        
        for idx in indices:
            # Find all cam files for this index
            cam_pattern = str(self.cam_dir / f"{idx}_cam_*.png")
            cam_files = glob.glob(cam_pattern)
            
            for cam_file in cam_files:
                filename = os.path.basename(cam_file)
                dest_path = output_dir / filename
                shutil.copy2(cam_file, dest_path)
                copied_count += 1
        
        print(f"Copied {copied_count} camera files to {output_dir}")
    
    def copy_lidar_files(self, indices, output_dir, split_name):
        """Copy lidar files for given indices"""
        print(f"\nCopying {split_name} lidar files...")
        copied_count = 0
        
        for idx in indices:
            # Copy both .pcd and .png lidar files
            lidar_patterns = [
                str(self.lidar_dir / f"{idx}_lidars_lidar.pcd"),
                str(self.lidar_dir / f"{idx}_lidars_ppi.png")
            ]
            
            for pattern in lidar_patterns:
                files = glob.glob(pattern)
                for lidar_file in files:
                    filename = os.path.basename(lidar_file)
                    dest_path = output_dir / filename
                    shutil.copy2(lidar_file, dest_path)
                    copied_count += 1
        
        print(f"Copied {copied_count} lidar files to {output_dir}")
    
    def copy_annotation_files(self, indices, output_dir, split_name):
        """Copy annotation files for given indices"""
        print(f"\nCopying {split_name} annotation files...")
        copied_count = 0
        
        for idx in indices:
            # Copy both 2dbox and 3dbox annotation files
            annotation_patterns = [
                str(self.annotation_dir / f"{idx}_ppi_2dbox.yaml"),
                str(self.annotation_dir / f"{idx}_pcd_3dbox.yaml")
            ]
            
            for pattern in annotation_patterns:
                files = glob.glob(pattern)
                for ann_file in files:
                    filename = os.path.basename(ann_file)
                    dest_path = output_dir / filename
                    shutil.copy2(ann_file, dest_path)
                    copied_count += 1
        
        print(f"Copied {copied_count} annotation files to {output_dir}")
    
    def copy_gt_files(self, indices, source_dir, output_dir, split_name, file_extensions):
        """Copy ground truth files from a source directory for given indices"""
        print(f"\nCopying {split_name} {source_dir.name} files...")
        copied_count = 0
        
        for idx in indices:
            # Copy files with different extensions
            for ext in file_extensions:
                pattern = str(source_dir / f"{idx}*{ext}")
                files = glob.glob(pattern)
                for gt_file in files:
                    filename = os.path.basename(gt_file)
                    dest_path = output_dir / filename
                    shutil.copy2(gt_file, dest_path)
                    copied_count += 1
        
        print(f"Copied {copied_count} {source_dir.name} files to {output_dir}")
    
    def copy_gt_meta_files(self, indices, output_dir, split_name):
        """Copy gt_meta files for given indices"""
        if self.gt_meta_dir.exists():
            self.copy_gt_files(indices, self.gt_meta_dir, output_dir, split_name, 
                             ['.yaml', '.yml', '.json', '.txt'])
        else:
            print(f"Warning: gt_meta directory not found: {self.gt_meta_dir}")
    
    def copy_gt_under_ego_files(self, indices, output_dir, split_name):
        """Copy gt_under_ego files for given indices"""
        if self.gt_under_ego_dir.exists():
            self.copy_gt_files(indices, self.gt_under_ego_dir, output_dir, split_name, 
                             ['.yaml', '.yml', '.json', '.txt'])
        else:
            print(f"Warning: gt_under_ego directory not found: {self.gt_under_ego_dir}")
    
    def copy_gt_under_world_files(self, indices, output_dir, split_name):
        """Copy gt_under_world files for given indices"""
        if self.gt_under_world_dir.exists():
            self.copy_gt_files(indices, self.gt_under_world_dir, output_dir, split_name, 
                             ['.yaml', '.yml', '.json', '.txt'])
        else:
            print(f"Warning: gt_under_world directory not found: {self.gt_under_world_dir}")
    
    def create_file_lists(self, train_indices, val_indices):
        """Create text files listing the indices for each split"""
        # Save train indices
        train_list_file = self.output_dir / "train_indices.txt"
        with open(train_list_file, 'w') as f:
            for idx in train_indices:
                f.write(f"{idx}\n")
        
        # Save val indices
        val_list_file = self.output_dir / "val_indices.txt"
        with open(val_list_file, 'w') as f:
            for idx in val_indices:
                f.write(f"{idx}\n")
        
        print(f"\nSaved train indices to: {train_list_file}")
        print(f"Saved validation indices to: {val_list_file}")
    
    def split_dataset(self):
        """Main function to split the dataset"""
        print("Starting dataset split...")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Use existing indices: {self.use_existing_indices}")
        if not self.use_existing_indices:
            print(f"Train ratio: {self.train_ratio}")
        
        # Check if source directories exist
        if not self.cam_dir.exists():
            raise FileNotFoundError(f"Camera directory not found: {self.cam_dir}")
        if not self.lidar_dir.exists():
            raise FileNotFoundError(f"Lidar directory not found: {self.lidar_dir}")
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")
        
        # Create output directories
        self.create_output_directories()
        
        # Get train and validation indices
        if self.use_existing_indices:
            # Read from existing files
            train_indices, val_indices = self.read_indices_from_files()
        else:
            # Generate new split
            indices = self.get_sample_indices()
            if not indices:
                raise ValueError("No sample indices found in the dataset")
            train_indices, val_indices = self.split_indices(indices)
        
        # Copy files for training set
        self.copy_cam_files(train_indices, self.train_cam_dir, "training")
        self.copy_lidar_files(train_indices, self.train_lidar_dir, "training")
        self.copy_annotation_files(train_indices, self.train_annotation_dir, "training")
        self.copy_gt_meta_files(train_indices, self.train_gt_meta_dir, "training")
        self.copy_gt_under_ego_files(train_indices, self.train_gt_under_ego_dir, "training")
        self.copy_gt_under_world_files(train_indices, self.train_gt_under_world_dir, "training")
        
        # Copy files for validation set
        self.copy_cam_files(val_indices, self.val_cam_dir, "validation")
        self.copy_lidar_files(val_indices, self.val_lidar_dir, "validation")
        self.copy_annotation_files(val_indices, self.val_annotation_dir, "validation")
        self.copy_gt_meta_files(val_indices, self.val_gt_meta_dir, "validation")
        self.copy_gt_under_ego_files(val_indices, self.val_gt_under_ego_dir, "validation")
        self.copy_gt_under_world_files(val_indices, self.val_gt_under_world_dir, "validation")
        
        # Create file lists (only if not using existing indices)
        if not self.use_existing_indices:
            self.create_file_lists(train_indices, val_indices)
        
        print(f"\n=== Dataset Split Complete ===")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Output saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/2025-10-18_22-26-46",
        help="Base directory containing the original dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/",
        help="Output directory for split dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (0.0 to 1.0), default: 0.8"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits, default: 42"
    )
    parser.add_argument(
        "--use_existing_indices",
        action="store_true",
        help="Use existing train_indices.txt and val_indices.txt files instead of generating new split"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0.0 and 1.0")
    
    # Initialize splitter and run
    splitter = DatasetSplitter(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_existing_indices=args.use_existing_indices
    )
    
    try:
        splitter.split_dataset()
    except Exception as e:
        print(f"Error during dataset split: {e}")
        raise


if __name__ == "__main__":
    main()