"""
SCIN Dataset Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from base_processor import BaseProcessor


class SCINProcessor(BaseProcessor):
    """Processor for SCIN dataset metadata."""
    
    def __init__(self):
        super().__init__('SCIN')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process SCIN dataset metadata."""
        
        # Look for the metadata files
        cases_file = dataset_path / 'scin_cases.csv'
        labels_file = dataset_path / 'scin_labels.csv'
        
        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Read the metadata files
        cases_df = self.read_csv_safely(cases_file)
        labels_df = self.read_csv_safely(labels_file)
        
        # Merge cases with labels
        merged_df = cases_df.merge(labels_df, on='case_id', how='inner')
        
        processed_samples = []
        
        for _, row in merged_df.iterrows():
            # Create image path (use the first image if multiple)
            image_path = row.get('image_1_path', '')
            
            # Extract demographics
            age_group = row.get('age_group', '')
            sex = row.get('sex_at_birth', '')
            
            # Extract body parts
            body_parts = []
            body_part_columns = [col for col in row.index if col.startswith('body_parts_')]
            for col in body_part_columns:
                if row[col] == 'YES':
                    body_parts.append(col.replace('body_parts_', ''))
            
            # Extract symptoms
            symptoms = []
            symptom_columns = [col for col in row.index if col.startswith('condition_symptoms_')]
            for col in symptom_columns:
                if row[col] == 'YES':
                    symptoms.append(col.replace('condition_symptoms_', ''))
            
            # Extract other symptoms
            other_symptoms = []
            other_symptom_columns = [col for col in row.index if col.startswith('other_symptoms_')]
            for col in other_symptom_columns:
                if row[col] == 'YES':
                    other_symptoms.append(col.replace('other_symptoms_', ''))
            
            # Extract additional metadata
            additional_metadata = {
                'case_id': row['case_id'],
                'source': row.get('source'),
                'year': row.get('year'),
                'age_group': age_group,
                'fitzpatrick_skin_type': row.get('fitzpatrick_skin_type'),
                'race_ethnicity': row.get('combined_race', ''),
                'body_parts': body_parts,
                'symptoms': symptoms,
                'other_symptoms': other_symptoms,
                'related_category': row.get('related_category'),
                'condition_duration': row.get('condition_duration'),
                'image_2_path': row.get('image_2_path'),
                'image_3_path': row.get('image_3_path'),
                'image_1_shot_type': row.get('image_1_shot_type'),
                'image_2_shot_type': row.get('image_2_shot_type'),
                'image_3_shot_type': row.get('image_3_shot_type')
            }
            
            # Add label information if available
            if 'label_name' in row:
                additional_metadata['label_name'] = row['label_name']
            if 'label_category' in row:
                additional_metadata['label_category'] = row['label_category']
            
            # Create standardized sample
            sample = self.create_standardized_sample(
                sample_id=str(row['case_id']),
                image_path=image_path,
                diagnosis=row.get('related_category', 'unknown'),
                age=self.parse_age_group(age_group),
                sex=sex,
                anatomical_site=', '.join(body_parts) if body_parts else None,
                additional_metadata=additional_metadata
            )
            
            # Add VQA questions
            sample['vqa_questions'] = self.generate_vqa_questions(sample)
            
            # Add SCIN-specific questions
            if body_parts:
                sample['vqa_questions'].append({
                    'question': 'Which body parts are affected?',
                    'answer': ', '.join(body_parts)
                })
            
            if symptoms:
                sample['vqa_questions'].append({
                    'question': 'What symptoms does the patient experience?',
                    'answer': ', '.join(symptoms)
                })
            
            if other_symptoms:
                sample['vqa_questions'].append({
                    'question': 'What other symptoms does the patient experience?',
                    'answer': ', '.join(other_symptoms)
                })
            
            if row.get('condition_duration'):
                sample['vqa_questions'].append({
                    'question': 'How long has the patient had this condition?',
                    'answer': row['condition_duration']
                })
            
            if row.get('fitzpatrick_skin_type'):
                sample['vqa_questions'].append({
                    'question': 'What is the Fitzpatrick skin type?',
                    'answer': f"Fitzpatrick type {row['fitzpatrick_skin_type']}"
                })
            
            processed_samples.append(sample)
        
        return processed_samples
    
    def parse_age_group(self, age_group: str) -> float:
        """Parse age group string to approximate age."""
        if not age_group or pd.isna(age_group):
            return None
        
        age_group = str(age_group).lower()
        
        # Map age groups to approximate ages
        age_mappings = {
            'age_0_9': 5,
            'age_10_19': 15,
            'age_20_29': 25,
            'age_30_39': 35,
            'age_40_49': 45,
            'age_50_59': 55,
            'age_60_69': 65,
            'age_70_79': 75,
            'age_80_89': 85,
            'age_90_99': 95,
            'age_unknown': None
        }
        
        return age_mappings.get(age_group, None) 