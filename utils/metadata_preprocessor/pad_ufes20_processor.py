"""
PAD-UFES20 Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class PADUfes20Processor(BaseProcessor):
    """Processor for PAD-UFES20 dataset metadata."""
    
    def __init__(self):
        super().__init__('PAD-UFES20')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process PAD-UFES20 dataset metadata."""
        
        # Look for the metadata file
        metadata_file = dataset_path / 'metadata.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read the metadata file
        df = self.read_csv_safely(metadata_file)
        
        processed_samples = []
        
        for _, row in df.iterrows():
            # Create image path
            image_path = f"images/{row['img_id']}"
            
            # Extract additional metadata
            additional_metadata = {
                'patient_id': row['patient_id'],
                'lesion_id': row['lesion_id'],
                'smoke': row.get('smoke'),
                'drink': row.get('drink'),
                'background_father': row.get('background_father'),
                'background_mother': row.get('background_mother'),
                'pesticide': row.get('pesticide'),
                'skin_cancer_history': row.get('skin_cancer_history'),
                'cancer_history': row.get('cancer_history'),
                'has_piped_water': row.get('has_piped_water'),
                'has_sewage_system': row.get('has_sewage_system'),
                'fitzpatrick': row.get('fitspatrick'),
                'region': row.get('region'),
                'diameter_1': row.get('diameter_1'),
                'diameter_2': row.get('diameter_2'),
                'itch': row.get('itch'),
                'grew': row.get('grew'),
                'hurt': row.get('hurt'),
                'changed': row.get('changed'),
                'bleed': row.get('bleed'),
                'elevation': row.get('elevation'),
                'biopsed': row.get('biopsed')
            }
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=row['img_id'],
                image_path=image_path,
                diagnosis=row['diagnostic'],
                age=row['age'],
                sex=row['gender'],
                anatomical_site=row['region'],
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add PAD-UFES20 specific questions
            if pd.notna(row.get('smoke')):
                sample['vqa_questions'].append({
                    'question': 'Does the patient smoke?',
                    'answer': 'Yes' if row['smoke'] else 'No'
                })
            
            if pd.notna(row.get('drink')):
                sample['vqa_questions'].append({
                    'question': 'Does the patient drink alcohol?',
                    'answer': 'Yes' if row['drink'] else 'No'
                })
            
            if pd.notna(row.get('skin_cancer_history')):
                sample['vqa_questions'].append({
                    'question': 'Does the patient have a history of skin cancer?',
                    'answer': 'Yes' if row['skin_cancer_history'] else 'No'
                })
            
            if pd.notna(row.get('fitspatrick')):
                sample['vqa_questions'].append({
                    'question': 'What is the Fitzpatrick skin type?',
                    'answer': f"Fitzpatrick type {row['fitspatrick']}"
                })
            
            # Add symptom-related questions
            symptoms = ['itch', 'grew', 'hurt', 'changed', 'bleed']
            for symptom in symptoms:
                if pd.notna(row.get(symptom)):
                    answer = 'Yes' if row[symptom] else 'No'
                    sample['vqa_questions'].append({
                        'question': f'Does the lesion {symptom}?',
                        'answer': answer
                    })
            
            processed_samples.append(sample)
        
        return processed_samples 