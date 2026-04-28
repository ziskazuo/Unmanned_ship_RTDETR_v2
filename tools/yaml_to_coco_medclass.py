#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML to COCO Format Converter (Full Name as Category)

This script converts YAML annotation files to COCO format JSON using full object names as categories.
Each unique object name becomes a separate category.
For example: USV_HouseBoat01_C_124 is treated as its own category.

Usage:
    python yaml_to_coco_fullname.py --yaml_dir ./annotations --output ./annotations.json
"""

import os
import json
import yaml
import glob
from datetime import datetime
import argparse
from collections import defaultdict


class YamlToCocoFullName:
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
        
        # Category management - using full names
        self.category_name_to_id = {}
        self.category_id_counter = 1
        self.category_stats = defaultdict(int)
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Full-Name Category Ship Detection Dataset",
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

    def normalize_category_name(self, object_name):
        """
        Normalize category name (keep full name but clean it up)
        
        Args:
            object_name (str): Full object name from YAML
            
        Returns:
            str: Normalized category name
        """
        # Keep the full name as is, just strip whitespace
        category = object_name.strip()
        
        # Optional: you can add some normalization rules here if needed
        # For example, convert to uppercase or replace certain characters
        # category = category.upper()  # Uncomment if you want uppercase
        
        return category

    def get_or_create_category_id(self, category_name):
        """
        Get existing category ID or create new one
        
        Args:
            category_name (str): Full category name
            
        Returns:
            int: Category ID
        """
        # Extract base type for category name
        extracted_category = "ship"  # Default category
        if "HouseBoat" in category_name:
            extracted_category = "HouseBoat"
        elif "Sailboat" in category_name:
            extracted_category = "Sailboat"
        elif "Yacht" in category_name:
            extracted_category = "Yacht"
        elif "Yachat" in category_name:
            extracted_category = "Yacht"  # Assume Yachat is a typo for Yacht
        
        if extracted_category not in self.category_name_to_id:
            category_id = self.category_id_counter
            self.category_name_to_id[extracted_category] = category_id
            
            # Add to COCO categories - use extracted category as name, supercategory as None
            category_info = {
                "id": category_id,
                "name": extracted_category,
                "supercategory": None
            }
            self.coco_format["categories"].append(category_info)
            
            self.category_id_counter += 1
            print(f"Created new category: {extracted_category} (ID: {category_id})")
        
        return self.category_name_to_id[extracted_category]

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
            
            # Use full object name as category
            category_name = self.normalize_category_name(obj_name)
            # Extract supercategory for stats and id
            extracted_category = "ship"
            if "HouseBoat" in category_name:
                extracted_category = "HouseBoat"
            elif "Sailboat" in category_name:
                extracted_category = "Sailboat"
            elif "Yacht" in category_name:
                extracted_category = "Yacht"
            elif "Yachat" in category_name:
                extracted_category = "Yacht"
            category_id = self.get_or_create_category_id(category_name)
            # Update statistics by supercategory
            self.category_stats[extracted_category] += 1
            
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
                categories_in_image = set(ann["category_id"] for ann in annotations)
                print(f"Processed {yaml_basename}: {len(annotations)} objects, {len(categories_in_image)} categories")
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
            print(f"Total unique categories: {len(self.coco_format['categories'])}")
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
        print(f"Total unique categories: {len(self.coco_format['categories'])}")
        print(f"Average objects per image: {total_annotations / total_images if total_images > 0 else 0:.2f}")
        
        # Count images with and without annotations
        images_with_objects = len(set(ann["image_id"] for ann in self.coco_format["annotations"]))
        images_without_objects = total_images - images_with_objects
        
        print(f"Images with objects: {images_with_objects}")
        print(f"Images without objects: {images_without_objects}")
        
        # Print category statistics (sorted by count)
        print(f"\n=== Category Statistics (Top 20) ===")
        sorted_categories = sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (cat_name, count) in enumerate(sorted_categories[:20]):
            category_id = self.category_name_to_id[cat_name]
            print(f"{i+1:2d}. {cat_name} (ID: {category_id}): {count} objects")
        if len(sorted_categories) > 20:
            print(f"... and {len(sorted_categories) - 20} more categories")
        
        # Print supercategory summary
        print(f"\n=== Supercategory Summary ===")
        supercategory_count = defaultdict(int)
        for category in self.coco_format["categories"]:
            supercategory_count[category["supercategory"]] += 1
        
        for super_cat, count in sorted(supercategory_count.items()):
            print(f"{super_cat}: {count} categories")

    def save_category_mapping(self):
        """
        Save category mapping to a separate file for reference
        """
        mapping_file = self.output_file.replace('.json', '_categories.json')
        
        category_mapping = {
            "total_categories": len(self.coco_format["categories"]),
            "categories": self.coco_format["categories"],
            "category_stats": dict(self.category_stats),
            "name_to_id": self.category_name_to_id,
            "sorted_by_frequency": sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
        }
        
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(category_mapping, f, indent=2, ensure_ascii=False)
            print(f"Category mapping saved to: {mapping_file}")
        except Exception as e:
            print(f"Error saving category mapping: {e}")

    def save_category_list(self):
        """
        Save a simple text file with all category names for reference
        """
        list_file = self.output_file.replace('.json', '_category_list.txt')
        
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                f.write("# Category List (Full Names)\n")
                f.write(f"# Total: {len(self.coco_format['categories'])} categories\n")
                f.write("# Format: ID | Name | Count | Supercategory\n\n")
                
                sorted_categories = sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
                for category_name, count in sorted_categories:
                    category_id = self.category_name_to_id[category_name]
                    # Find supercategory
                    supercategory = "ship"
                    for cat in self.coco_format["categories"]:
                        if cat["name"] == category_name:
                            supercategory = cat["supercategory"]
                            break
                    
                    f.write(f"{category_id:3d} | {category_name:<30} | {count:4d} | {supercategory}\n")
            
            print(f"Category list saved to: {list_file}")
        except Exception as e:
            print(f"Error saving category list: {e}")

    def save_label_file(self):
        """
        Save label_list.txt file with unique category names (超类别)
        """
        label_file = self.output_file.replace('.json', '_label_list.txt')
        try:
            with open(label_file, 'w', encoding='utf-8') as f:
                # Get unique category names (即超类别)
                names = set()
                for category in self.coco_format["categories"]:
                    if category['name'] is not None:
                        names.add(category['name'])
                sorted_names = sorted(names)
                for name in sorted_names:
                    f.write(f"{name}\n")
            print(f"Label file saved to: {label_file}")
            print(f"Saved {len(sorted_names)} categories: {', '.join(sorted_names)}")
        except Exception as e:
            print(f"Error saving label file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert YAML annotations to COCO format using full names as categories")
    parser.add_argument(
        "--yaml_dir", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset_one_medclass/train/annotations",
        help="Directory containing YAML files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/dataset_one_medclass/train_coco_annotations.json",
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
    print(f"Starting full-name category conversion...\n")
    
    # Initialize converter and run conversion
    converter = YamlToCocoFullName(
        yaml_dir=args.yaml_dir,
        output_file=args.output,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    converter.convert()
    converter.print_statistics()
    #converter.save_category_mapping()
    #converter.save_category_list()
    converter.save_label_file()


if __name__ == "__main__":
    main()