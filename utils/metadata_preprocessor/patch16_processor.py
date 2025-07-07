"""
Patch16 Dataset Processor
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class Patch16Processor(BaseProcessor):
    """Processor for Patch16 dataset metadata."""
    
    def __init__(self):
        super().__init__('Patch16')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process Patch16 dataset metadata."""
        
        # Look for the metadata files
        tiles_file = dataset_path / 'tiles-v2.csv'
        class_dict_file = dataset_path / 'class_dict.json'
        
        if not tiles_file.exists():
            raise FileNotFoundError(f"Tiles file not found: {tiles_file}")
        if not class_dict_file.exists():
            raise FileNotFoundError(f"Class dict file not found: {class_dict_file}")
        
        # Read the metadata files
        df = self.read_csv_safely(tiles_file)
        
        # Read class dictionary
        with open(class_dict_file, 'r') as f:
            class_dict = json.load(f)
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row.get('tile_id', row.get('filename', 'unknown'))}"
            
            # Map class number to diagnosis
            class_num = row.get('class', 0)
            diagnosis = class_dict.get(str(class_num), 'unknown')
            
            # Extract additional metadata
            additional_metadata = {
                'tile_id': row.get('tile_id'),
                'class_num': class_num,
                'class_dict': class_dict,
                'original_data': dict(row)
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=str(row.get('tile_id', len(processed_samples))),
                image_path=image_path,
                diagnosis=diagnosis,
                age=row.get('age'),
                sex=row.get('sex', row.get('gender')),
                anatomical_site=row.get('anatomical_site', row.get('location')),
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add Patch16-specific questions
            sample['vqa_questions'].append({
                'question': 'What is the class number for this patch?',
                'answer': str(class_num)
            })
            
            processed_samples.append(sample)
        
        return processed_samples 