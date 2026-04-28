#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML to COCO Format Converter (Multi-Class Version)

This script converts YAML annotation files to COCO format JSON with multiple ship categories.
Categories are determined from object names in YAML files.
For example: USV_HouseBoat01_C_124 -> HouseBoat category

Usage:
    python yaml_to_coco_multiclass.py --yaml_dir ./annotations --output ./annotations.json
"""

import os
import json
import yaml
import glob
import re
from datetime import datetime
import argparse
from collections import defaultdict


class YamlToCocoMultiClass:
    def __init__(self, yaml_dir, output_file, image_width=1000, image_height=1000):
        """
        Initialize the converter
        
        Args:
            yaml_dir (str): Directory containing YAML files
            output_file (str): Output COCO JSON file path
            image_width (int): Default image width
            image_height (int): Default image height
        """
        self.yaml_dir = yaml_dir
        self.output_file = output_file
        self.image_width = image_width
        self.image_height = image_height
        
        # Category management
        self.category_name_to_id = {}
        self.category_id_counter = 1
        self.category_stats = defaultdict(int)
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Multi-Class Ship Detection Dataset",
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
            "categories": [],  # Will be populated dynamically
            "images": [],
            "annotations": []
        }
        
        self.image_id_counter = 0
        self.annotation_id_counter = 0

    def extract_category_from_name(self, object_name):
        """
        Extract category from object name
        
        Examples:
            USV_HouseBoat01_C_124 -> HouseBoat
            USV_Sailboat03_C_124 -> Sailboat
            USV_Yacht02_C_124 -> Yacht
            USV_Yachat04_C_124 -> Yachat (keep as is if unusual spelling)
        
        Args:
            object_name (str): Object name from YAML
            
        Returns:
            str: Extracted category name
        """
        # Remove USV_ prefix if exists
        name = object_name
        if name.startswith('USV_'):
            name = name[4:]
        
        # Extract the main category part before numbers and suffixes
        # Pattern: find letters at the beginning, stop at first digit or underscore
        match = re.match(r'^([A-Za-z]+)', name)
        if match:
            category = match.group(1)
        else:
            # Fallback: use the whole name
            category = name.split('_')[0] if '_' in name else name
        
        # Normalize category name
        category = category.lower().capitalize()
        
        return category

    def get_or_create_category_id(self, category_name):
        """
        Get existing category ID or create new one (use full object name as category)
        
        Args:
            category_name (str): Full object name from YAML
        Returns:
            int: Category ID
        """
        if category_name not in self.category_name_to_id:
            category_id = self.category_id_counter
            self.category_name_to_id[category_name] = category_id
            # Add to COCO categories: name为完整对象名，supercategory统一为'ship'
            category_info = {
                "id": category_id,
                "name": category_name,
                "supercategory": "ship"
            }
            self.coco_format["categories"].append(category_info)
            self.category_id_counter += 1
            print(f"Created new category: {category_name} (ID: {category_id})")
        return self.category_name_to_id[category_name]

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
            # 用完整对象名作为类别
            category_name = obj_name.strip()
            category_id = self.get_or_create_category_id(category_name)
            # Update statistics
            self.category_stats[category_name] += 1
            poly_points = obj_data['poly']
            segmentation, bbox, area = self.parse_polygon(poly_points)
            if segmentation is None or bbox is None:
                print(f"Skipping invalid polygon in {yaml_file} for object {obj_name}")
                continue
            annotation = {
                "id": self.annotation_id_counter,
                "image_id": self.image_id_counter,
                "category_id": category_id,
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
            # Convert 0000_ppi_2dbox.yaml to 0000.png
            image_name = yaml_basename.replace('_ppi_2dbox.yaml', '.png')
            
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
            print(f"Total categories: {len(self.coco_format['categories'])}")
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
        print(f"Total categories: {len(self.coco_format['categories'])}")
        print(f"Average objects per image: {total_annotations / total_images if total_images > 0 else 0:.2f}")
        
        # Count images with and without annotations
        images_with_objects = len(set(ann["image_id"] for ann in self.coco_format["annotations"]))
        images_without_objects = total_images - images_with_objects
        
        print(f"Images with objects: {images_with_objects}")
        print(f"Images without objects: {images_without_objects}")
        
        # Print category statistics
        print(f"\n=== Category Statistics ===")
        for category_name, count in sorted(self.category_stats.items()):
            category_id = self.category_name_to_id[category_name]
            print(f"{category_name} (ID: {category_id}): {count} objects")
        
        # Print category list for reference
        print(f"\n=== Category Mapping ===")
        for category in self.coco_format["categories"]:
            print(f"ID {category['id']}: {category['name']}")

    def save_category_mapping(self):
        """
        Save category mapping to a separate file for reference
        """
        mapping_file = self.output_file.replace('.json', '_categories.json')
        
        category_mapping = {
            "categories": self.coco_format["categories"],
            "category_stats": dict(self.category_stats),
            "name_to_id": self.category_name_to_id
        }
        
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(category_mapping, f, indent=2, ensure_ascii=False)
            print(f"Category mapping saved to: {mapping_file}")
        except Exception as e:
            print(f"Error saving category mapping: {e}")

    def save_label_file(self):
        """
        Save label_list.txt file with category names
        """
        label_file = self.output_file.replace('.json', '_label_list.txt')
        
        try:
            with open(label_file, 'w', encoding='utf-8') as f:
                # Sort categories by ID to maintain consistent order
                sorted_categories = sorted(self.coco_format["categories"], key=lambda x: x["id"])
                for category in sorted_categories:
                    f.write(f"{category['name']}\n")
            
            print(f"Label file saved to: {label_file}")
        except Exception as e:
            print(f"Error saving label file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert YAML annotations to COCO format with multi-class support")
    parser.add_argument(
        "--yaml_dir", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset_one_allclass/val/annotations",
        help="Directory containing YAML files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset_one_allclass/val_coco_annotations.json",
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
    print(f"Starting multi-class conversion...\n")
    
    # Initialize converter and run conversion
    converter = YamlToCocoMultiClass(
        yaml_dir=args.yaml_dir,
        output_file=args.output,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    converter.convert()
    converter.print_statistics()
    #converter.save_category_mapping()
    converter.save_label_file()


if __name__ == "__main__":
    main()