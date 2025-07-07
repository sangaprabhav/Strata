"""
Base processor class for dermatology dataset metadata preprocessing.
"""

import json
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional


class BaseProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.required_fields = [
            'dataset_name',
            'sample_id',
            'image_path',
            'diagnosis',
            'age',
            'sex',
            'anatomical_site',
            'metadata'
        ]
    
    @abstractmethod
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset and return standardized metadata."""
        pass
    
    def create_standardized_sample(self, 
                                   sample_id: str,
                                   image_path: str,
                                   diagnosis: str,
                                   age: Optional[float] = None,
                                   sex: Optional[str] = None,
                                   anatomical_site: Optional[str] = None,
                                   additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a standardized sample dictionary."""
        
        sample = {
            'dataset_name': self.dataset_name,
            'sample_id': sample_id,
            'image_path': image_path,
            'diagnosis': self.standardize_diagnosis(diagnosis),
            'age': self.standardize_age(age),
            'sex': self.standardize_sex(sex),
            'anatomical_site': self.standardize_anatomical_site(anatomical_site),
            'metadata': additional_metadata or {}
        }
        
        return sample
    
    def standardize_diagnosis(self, diagnosis: str) -> str:
        """Standardize diagnosis labels."""
        if not diagnosis or pd.isna(diagnosis):
            return 'unknown'
        
        diagnosis = str(diagnosis).lower().strip()
        
        # Common diagnosis mappings
        diagnosis_map = {
            'mel': 'melanoma',
            'melanoma': 'melanoma',
            'melanoma-in-situ': 'melanoma',
            'malignant melanoma': 'melanoma',
            'nv': 'nevus',
            'nevus': 'nevus',
            'benign': 'benign',
            'bkl': 'benign_keratosis',
            'benign keratosis': 'benign_keratosis',
            'bcc': 'basal_cell_carcinoma',
            'basal cell carcinoma': 'basal_cell_carcinoma',
            'akiec': 'actinic_keratosis',
            'actinic keratosis': 'actinic_keratosis',
            'df': 'dermatofibroma',
            'dermatofibroma': 'dermatofibroma',
            'vasc': 'vascular_lesion',
            'vascular lesion': 'vascular_lesion',
            'scc': 'squamous_cell_carcinoma',
            'squamous cell carcinoma': 'squamous_cell_carcinoma',
            'squamous-cell-carcinoma-in-situ': 'squamous_cell_carcinoma'
        }
        
        return diagnosis_map.get(diagnosis, diagnosis)
    
    def standardize_age(self, age: Any) -> Optional[float]:
        """Standardize age values."""
        if age is None or pd.isna(age):
            return None
        
        try:
            age_val = float(age)
            return age_val if 0 <= age_val <= 150 else None
        except (ValueError, TypeError):
            return None
    
    def standardize_sex(self, sex: Any) -> Optional[str]:
        """Standardize sex values."""
        if not sex or pd.isna(sex):
            return None
        
        sex = str(sex).lower().strip()
        
        sex_map = {
            'male': 'male',
            'female': 'female',
            'm': 'male',
            'f': 'female',
            'man': 'male',
            'woman': 'female'
        }
        
        return sex_map.get(sex, 'unknown')
    
    def standardize_anatomical_site(self, site: Any) -> Optional[str]:
        """Standardize anatomical site values."""
        if not site or pd.isna(site):
            return None
        
        site = str(site).lower().strip()
        
        # Common anatomical site mappings
        site_map = {
            'face': 'head_neck',
            'head': 'head_neck',
            'neck': 'head_neck',
            'head/neck': 'head_neck',
            'scalp': 'head_neck',
            'ear': 'head_neck',
            'back': 'trunk',
            'chest': 'trunk',
            'abdomen': 'trunk',
            'trunk': 'trunk',
            'anterior torso': 'trunk',
            'posterior torso': 'trunk',
            'arm': 'upper_extremity',
            'hand': 'upper_extremity',
            'palm': 'upper_extremity',
            'back of hand': 'upper_extremity',
            'upper extremity': 'upper_extremity',
            'leg': 'lower_extremity',
            'foot': 'lower_extremity',
            'foot top or side': 'lower_extremity',
            'foot sole': 'lower_extremity',
            'lower extremity': 'lower_extremity',
            'genital': 'genitalia',
            'genitalia': 'genitalia',
            'groin': 'genitalia',
            'genitalia or groin': 'genitalia',
            'buttocks': 'trunk'
        }
        
        return site_map.get(site, site)
    
    def read_csv_safely(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Safely read CSV files with error handling."""
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def generate_vqa_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate VQA questions for a sample."""
        questions = []
        
        # Basic questions
        questions.append({
            'question': 'What is the diagnosis of this skin lesion?',
            'answer': sample.get('diagnosis', 'unknown')
        })
        
        if sample.get('age'):
            questions.append({
                'question': 'What is the age of the patient?',
                'answer': f"{sample['age']} years old"
            })
        
        if sample.get('sex'):
            questions.append({
                'question': 'What is the sex of the patient?',
                'answer': sample['sex']
            })
        
        if sample.get('anatomical_site'):
            questions.append({
                'question': 'Where is this lesion located on the body?',
                'answer': sample['anatomical_site']
            })
        
        # Additional diagnostic questions
        if sample.get('diagnosis') == 'melanoma':
            questions.append({
                'question': 'Is this lesion malignant?',
                'answer': 'Yes, this is a malignant melanoma'
            })
        elif sample.get('diagnosis') in ['nevus', 'benign']:
            questions.append({
                'question': 'Is this lesion malignant?',
                'answer': 'No, this is a benign lesion'
            })
        
        return questions 