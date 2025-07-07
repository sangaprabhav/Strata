"""
MRA-MIDAS Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class MRAMIDASProcessor(BaseProcessor):
    """Processor for MRA-MIDAS dataset metadata."""
    
    def __init__(self):
        super().__init__('MRA-MIDAS')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process MRA-MIDAS dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'release_midas.xlsx'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the Excel file
        try:
            df = pd.read_excel(metadata_file)
        except Exception as e:
            print(f"Error reading MRA-MIDAS metadata: {e}")
            return []
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row.get('image_id', row.get('filename', 'unknown'))}"
            
            # Extract additional metadata
            additional_metadata = {
                'original_data': dict(row)
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=str(row.get('id', len(processed_samples))),
                image_path=image_path,
                diagnosis=row.get('diagnosis', 'unknown'),
                age=row.get('age'),
                sex=row.get('sex', row.get('gender')),
                anatomical_site=row.get('anatomical_site', row.get('location')),
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            processed_samples.append(sample)
        
        return processed_samples 