#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML to COCO Format Converter

This script converts YAML annotation files to COCO format JSON.
Each YAML file contains object annotations with polygon coordinates.
All objects are labeled as 'ship' class.

Usage:
    python yaml_to_coco.py
"""

import os
import json
import yaml
import glob
from datetime import datetime
import argparse


class YamlToCoco:
    def __init__(self, yaml_dir, output_file, image_width=1000, image_height=1000):
        """
        Initialize the converter
        
        Args:
            yaml_dir (str): Directory containing YAML files
            output_file (str): Output COCO JSON file path
            image_width (int): Default image width (assuming all images have same size)
            image_height (int): Default image height (assuming all images have same size)
        """
        self.yaml_dir = yaml_dir
        self.output_file = output_file
        self.image_width = image_width
        self.image_height = image_height
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Ship Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Unmanned Ship RTDETR",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "ship",
                    "supercategory": "none"
                }
            ],
            "images": [],
            "annotations": []
        }
        
        self.image_id_counter = 0
        self.annotation_id_counter = 0
    def process_yaml_file(self, yaml_file):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML file {yaml_file}: {e}")
            return []
        
        if not data:  # 如果YAML文件为空或None
            return []
    def parse_polygon(self, poly_points):
        """
        Convert polygon points to segmentation format and calculate bbox
        
        Args:
            poly_points (list): List of [x, y] coordinates
            
        Returns:
            tuple: (segmentation, bbox, area)
        """
        if len(poly_points) < 3:
            return None, None, 0
            
        # Flatten polygon points for segmentation format
        segmentation = []
        x_coords = []
        y_coords = []
        
        for point in poly_points:
            x, y = point[0], point[1]
            segmentation.extend([float(x), float(y)])
            x_coords.append(float(x))
            y_coords.append(float(y))
        
        # Calculate bounding box [x, y, width, height]
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        
        # Calculate approximate area using shoelace formula
        area = self.calculate_polygon_area(poly_points)
        
        return [segmentation], bbox, area

    def calculate_polygon_area(self, points):
        """
        Calculate polygon area using shoelace formula
        
        Args:
            points (list): List of [x, y] coordinates
            
        Returns:
            float: Polygon area
        """
        if len(points) < 3:
            return 0
            
        area = 0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
            
        return abs(area) / 2.0

    def process_yaml_file(self, yaml_file):
        """
        Process a single YAML file and extract annotations
        
        Args:
            yaml_file (str): Path to YAML file
            
        Returns:
            list: List of annotation dictionaries
        """
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML file {yaml_file}: {e}")
            return []
        
        if not data:
            return []
        
        annotations = []
        
        for obj_name, obj_data in data.items():
            if 'poly' not in obj_data:
                continue
                
            poly_points = obj_data['poly']
            segmentation, bbox, area = self.parse_polygon(poly_points)
            
            if segmentation is None or bbox is None:
                print(f"Skipping invalid polygon in {yaml_file} for object {obj_name}")
                continue
            
            annotation = {
                "id": self.annotation_id_counter,
                "image_id": self.image_id_counter,
                "category_id": 1,  # 'ship' category
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            annotations.append(annotation)
            self.annotation_id_counter += 1
            
        return annotations

    def convert(self):
        """
        Convert all YAML files to COCO format
        """
        # Find all *_ppi_2dbox.yaml files
        yaml_pattern = os.path.join(self.yaml_dir, "*_ppi_2dbox.yaml")
        yaml_files = glob.glob(yaml_pattern)
        yaml_files.sort()
        
        print(f"Found {len(yaml_files)} YAML files to process")
        
        if not yaml_files:
            print(f"No YAML files found matching pattern: {yaml_pattern}")
            return
        
        processed_count = 0
        
        for yaml_file in yaml_files:
            # Extract image filename from YAML filename
            yaml_basename = os.path.basename(yaml_file)
            # Convert 0000_ppi_2dbox.yaml to 0000.jpg (or other image format)
            image_name = yaml_basename.replace('_ppi_2dbox.yaml', '.jpg')
            
            # Add image info
            image_info = {
                "id": self.image_id_counter,
                "width": self.image_width,
                "height": self.image_height,
                "file_name": image_name,
                "license": 1,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.coco_format["images"].append(image_info)
            
            # Process annotations for this image
            annotations = self.process_yaml_file(yaml_file)
            self.coco_format["annotations"].extend(annotations)
            
            if annotations:
                processed_count += 1
                print(f"Processed {yaml_basename}: {len(annotations)} objects")
            else:
                print(f"Processed {yaml_basename}: no objects found")
            
            self.image_id_counter += 1
        
        # Save COCO format JSON
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.coco_format, f, indent=2, ensure_ascii=False)
            
            print(f"\nConversion completed!")
            print(f"Total images processed: {len(self.coco_format['images'])}")
            print(f"Total annotations: {len(self.coco_format['annotations'])}")
            print(f"Output saved to: {self.output_file}")
            
        except Exception as e:
            print(f"Error saving COCO JSON file: {e}")

    def print_statistics(self):
        """
        Print conversion statistics
        """
        total_images = len(self.coco_format["images"])
        total_annotations = len(self.coco_format["annotations"])
        
        print(f"\n=== Conversion Statistics ===")
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"Average objects per image: {total_annotations / total_images if total_images > 0 else 0:.2f}")
        
        # Count images with and without annotations
        images_with_objects = len(set(ann["image_id"] for ann in self.coco_format["annotations"]))
        images_without_objects = total_images - images_with_objects
        
        print(f"Images with objects: {images_with_objects}")
        print(f"Images without objects: {images_without_objects}")

    def save_label_file(self):
        """
        Save label_list.txt file with ship category only
        """
        label_file = self.output_file.replace('.json', '_label_list.txt')
        
        try:
            with open(label_file, 'w', encoding='utf-8') as f:
                # In single-class version, only output "ship"
                f.write("ship\n")
            
            print(f"Label file saved to: {label_file}")
            print(f"Saved 1 category: ship")
        except Exception as e:
            print(f"Error saving label file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert YAML annotations to COCO format")
    parser.add_argument(
        "--yaml_dir", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/val/annotations",
        help="Directory containing YAML files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset/val/val_coco_annotations.json",
        help="Output COCO JSON file path"
    )
    parser.add_argument(
        "--image_width", 
        type=int, 
        default=1000,
        help="Default image width (default: 1000)"
    )
    parser.add_argument(
        "--image_height", 
        type=int, 
        default=1000,
        help="Default image height (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.yaml_dir):
        print(f"Error: Input directory does not exist: {args.yaml_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Input directory: {args.yaml_dir}")
    print(f"Output file: {args.output}")
    print(f"Image dimensions: {args.image_width} x {args.image_height}")
    print(f"Starting conversion...\n")
    
    # Initialize converter and run conversion
    converter = YamlToCoco(
        yaml_dir=args.yaml_dir,
        output_file=args.output,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    converter.convert()
    converter.print_statistics()
    converter.save_label_file()


if __name__ == "__main__":
    main()