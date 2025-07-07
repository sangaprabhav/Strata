"""
DDI2 Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class DDI2Processor(BaseProcessor):
    """Processor for DDI2 dataset metadata."""
    
    def __init__(self):
        super().__init__('DDI2')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process DDI2 dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'final_DDI2_Asian_spreadsheet.xlsx'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the Excel file
        try:
            df = pd.read_excel(metadata_file)
        except Exception as e:
            print(f"Error reading DDI2 metadata: {e}")
            return []
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path (assuming filename column exists)
            image_path = f"images/{row.get('filename', row.get('image_id', 'unknown'))}"
            
            # Extract additional metadata
            additional_metadata = {
                'original_data': dict(row)
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=str(row.get('id', row.get('image_id', len(processed_samples)))),
                image_path=image_path,
                diagnosis=row.get('diagnosis', row.get('disease', 'unknown')),
                age=row.get('age'),
                sex=row.get('sex', row.get('gender')),
                anatomical_site=row.get('anatomical_site', row.get('location')),
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            processed_samples.append(sample)
        
        return processed_samples 