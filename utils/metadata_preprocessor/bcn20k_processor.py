"""
BCN20K Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class BCN20KProcessor(BaseProcessor):
    """Processor for BCN20K dataset metadata."""
    
    def __init__(self):
        super().__init__('BCN20K')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process BCN20K dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'bcn20000_metadata_2025-05-22.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the metadata file with error handling for large files
        try:
            df = pd.read_csv(metadata_file, low_memory=False)
        except Exception as e:
            print(f"Error reading BCN20K metadata: {e}")
            return []
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row['isic_id']}.jpg"
            
            # Extract additional metadata
            additional_metadata = {
                'isic_id': row['isic_id'],
                'attribution': row.get('attribution'),
                'copyright_license': row.get('copyright_license'),
                'anatom_site_special': row.get('anatom_site_special'),
                'benign_malignant': row.get('benign_malignant'),
                'concomitant_biopsy': row.get('concomitant_biopsy'),
                'diagnosis_1': row.get('diagnosis_1'),
                'diagnosis_2': row.get('diagnosis_2'),
                'diagnosis_3': row.get('diagnosis_3'),
                'diagnosis_confirm_type': row.get('diagnosis_confirm_type'),
                'image_type': row.get('image_type'),
                'lesion_id': row.get('lesion_id'),
                'melanocytic': row.get('melanocytic')
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=row['isic_id'],
                image_path=image_path,
                diagnosis=row.get('diagnosis_1', 'unknown'),
                age=row.get('age_approx'),
                sex=row.get('sex'),
                anatomical_site=row.get('anatom_site_general'),
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add BCN20K-specific questions
            if pd.notna(row.get('benign_malignant')):
                sample['vqa_questions'].append({
                    'question': 'Is this lesion benign or malignant?',
                    'answer': row['benign_malignant']
                })
            
            if pd.notna(row.get('melanocytic')):
                melanocytic_answer = 'Yes' if row['melanocytic'] else 'No'
                sample['vqa_questions'].append({
                    'question': 'Is this lesion melanocytic?',
                    'answer': melanocytic_answer
                })
            
            if pd.notna(row.get('diagnosis_confirm_type')):
                sample['vqa_questions'].append({
                    'question': 'How was this diagnosis confirmed?',
                    'answer': row['diagnosis_confirm_type']
                })
            
            processed_samples.append(sample)
        
        return processed_samples 