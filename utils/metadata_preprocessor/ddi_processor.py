"""
DDI Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class DDIProcessor(BaseProcessor):
    """Processor for DDI dataset metadata."""
    
    def __init__(self):
        super().__init__('DDI')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process DDI dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'ddi_metadata.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the metadata file
        df = self.read_csv_safely(metadata_file)
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row['DDI_file']}"
            
            # Extract additional metadata
            additional_metadata = {
                'ddi_id': row['DDI_ID'],
                'skin_tone': row['skin_tone'],
                'malignant': row['malignant']
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=str(row['DDI_ID']),
                image_path=image_path,
                diagnosis=row['disease'],
                age=None,  # Age not available in DDI
                sex=None,  # Sex not available in DDI
                anatomical_site=None,  # Anatomical site not available in DDI
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add skin tone specific questions
            if row['skin_tone']:
                sample['vqa_questions'].append({
                    'question': 'What is the skin tone of this patient?',
                    'answer': f"Skin tone: {row['skin_tone']}"
                })
            
            # Add malignancy questions
            if pd.notna(row['malignant']):
                malignant_answer = 'Yes' if row['malignant'] else 'No'
                sample['vqa_questions'].append({
                    'question': 'Is this lesion malignant?',
                    'answer': malignant_answer
                })
            
            processed_samples.append(sample)
        
        return processed_samples 