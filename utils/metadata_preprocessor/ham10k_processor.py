"""
HAM10K Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class HAM10KProcessor(BaseProcessor):
    """Processor for HAM10K dataset metadata."""
    
    def __init__(self):
        super().__init__('HAM10K')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process HAM10K dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'HAM10000_metadata'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the metadata file (CSV with header)
        df = pd.read_csv(metadata_file)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'lesion_id': 'ham_id',
            'image_id': 'isic_id',
            'dx': 'diagnosis',
            'dx_type': 'diagnosis_type'
        })
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path (assuming images are in a subdirectory)
            image_path = f"images/{row['isic_id']}.jpg"
            
            # Map HAM10K diagnosis codes to standard format
            diagnosis_mapping = {
                'akiec': 'actinic_keratosis',
                'bcc': 'basal_cell_carcinoma',
                'bkl': 'benign_keratosis',
                'df': 'dermatofibroma',
                'mel': 'melanoma',
                'nv': 'nevus',
                'vasc': 'vascular_lesion'
            }
            
            # Map anatomical sites
            anatomical_mapping = {
                'abdomen': 'trunk',
                'back': 'trunk',
                'chest': 'trunk',
                'ear': 'head_neck',
                'face': 'head_neck',
                'foot': 'lower_extremity',
                'hand': 'upper_extremity',
                'lower leg': 'lower_extremity',
                'neck': 'head_neck',
                'scalp': 'head_neck',
                'trunk': 'trunk',
                'upper leg': 'lower_extremity',
                'acral': 'lower_extremity'  # Acral sites (hands/feet)
            }
            
            # Extract additional metadata
            additional_metadata = {
                'ham_id': row['ham_id'],
                'isic_id': row['isic_id'],
                'diagnosis_type': row['diagnosis_type'],
                'dataset_source': row['dataset'],
                'original_diagnosis': row['diagnosis'],
                'original_localization': row['localization']
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=row['ham_id'],
                image_path=image_path,
                diagnosis=diagnosis_mapping.get(row['diagnosis'], row['diagnosis']),
                age=row['age'],
                sex=row['sex'],
                anatomical_site=anatomical_mapping.get(row['localization'], row['localization']),
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            processed_samples.append(sample)
        
        return processed_samples 