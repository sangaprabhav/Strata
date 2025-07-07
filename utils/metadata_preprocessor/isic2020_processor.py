"""
ISIC2020 Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class ISIC2020Processor(BaseProcessor):
    """Processor for ISIC2020 dataset metadata."""
    
    def __init__(self):
        super().__init__('ISIC2020')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process ISIC2020 dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'ISIC_2020_Training_GroundTruth.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the metadata file
        df = self.read_csv_safely(metadata_file)
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row['image_name']}.jpg"
            
            # Extract additional metadata
            additional_metadata = {
                'patient_id': row['patient_id'],
                'benign_malignant': row['benign_malignant'],
                'target': row['target']
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=row['image_name'],
                image_path=image_path,
                diagnosis=row['diagnosis'],
                age=row['age_approx'],
                sex=row['sex'],
                anatomical_site=row['anatom_site_general_challenge'],
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add benign/malignant questions
            if pd.notna(row['benign_malignant']):
                sample['vqa_questions'].append({
                    'question': 'Is this lesion benign or malignant?',
                    'answer': row['benign_malignant']
                })
            
            # Add target-specific questions (for competition)
            if pd.notna(row['target']):
                target_answer = 'Yes' if row['target'] == 1 else 'No'
                sample['vqa_questions'].append({
                    'question': 'Is this lesion the target class for the competition?',
                    'answer': target_answer
                })
            
            processed_samples.append(sample)
        
        return processed_samples 