#!/usr/bin/env python3
"""
VQA Question Enricher for Dermatology Dataset
Generates additional diverse and clinically relevant questions for VQA training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional


class VQAEnricher:
    """Enriches VQA datasets with additional diverse questions."""
    
    def __init__(self):
        self.diagnosis_templates = {
            'melanoma': [
                "Is this lesion malignant?",
                "What type of skin cancer is this?",
                "Is this a melanoma?",
                "What is the most concerning diagnosis for this lesion?",
                "Should this lesion be biopsied?",
                "Is immediate medical attention needed for this lesion?",
                "What is the prognosis for this type of lesion?",
                "Is this lesion life-threatening?",
                "What treatment is typically needed for this diagnosis?"
            ],
            'nevus': [
                "Is this lesion benign?",
                "Is this a mole?",
                "What type of pigmented lesion is this?",
                "Is this lesion harmless?",
                "Does this lesion require treatment?",
                "Is this a common type of skin lesion?",
                "Should the patient monitor this lesion?",
                "Is this lesion likely to change over time?"
            ],
            'basal_cell_carcinoma': [
                "What type of skin cancer is this?",
                "Is this a non-melanoma skin cancer?",
                "Is this lesion malignant?",
                "What is the typical treatment for this lesion?",
                "Does this type of cancer metastasize?",
                "Is this a common type of skin cancer?",
                "What causes this type of lesion?"
            ],
            'squamous_cell_carcinoma': [
                "What type of skin cancer is this?",
                "Is this lesion malignant?",
                "Is this a non-melanoma skin cancer?",
                "What is the risk of metastasis?",
                "Is UV exposure a risk factor for this lesion?",
                "What treatment is needed?"
            ],
            'actinic_keratosis': [
                "Is this a precancerous lesion?",
                "What is the risk of malignant transformation?",
                "Is this related to sun exposure?",
                "Should this lesion be treated?",
                "Is this a common finding in sun-exposed areas?",
                "What is the significance of this lesion?"
            ]
        }
        
        self.anatomical_questions = {
            'head_neck': [
                "Where on the body is this lesion located?",
                "Is this lesion on sun-exposed skin?",
                "Is this a common location for skin cancer?",
                "What body region is affected?",
                "Is this location cosmetically sensitive?",
                "Would this location affect treatment planning?"
            ],
            'trunk': [
                "Where on the body is this lesion located?",
                "Is this lesion on the torso?",
                "What body region is affected?",
                "Is this a hidden area of the body?",
                "Would clothing typically cover this area?"
            ],
            'upper_extremity': [
                "Where on the body is this lesion located?",
                "Is this lesion on the arm or hand?",
                "What body region is affected?",
                "Is this a sun-exposed area?",
                "Is this a common location for skin lesions?"
            ],
            'lower_extremity': [
                "Where on the body is this lesion located?",
                "Is this lesion on the leg or foot?",
                "What body region is affected?",
                "Is this a pressure-bearing area?",
                "Could this location affect healing?"
            ]
        }
        
        self.demographic_questions = {
            'age': [
                "What age group does this patient belong to?",
                "Is this patient elderly?",
                "Is this a pediatric case?",
                "Is age a risk factor for this type of lesion?",
                "Does age affect the prognosis?",
                "Is this typical for this age group?"
            ],
            'sex': [
                "What is the patient's gender?",
                "Is there a gender predisposition for this condition?",
                "Does sex affect the risk of this lesion?",
                "Is this more common in males or females?"
            ]
        }
        
        self.clinical_questions = [
            "Should this lesion be monitored?",
            "Is a biopsy recommended?",
            "What follow-up is needed?",
            "Is dermatologist referral indicated?",
            "Should the patient be concerned?",
            "Is immediate treatment required?",
            "What is the urgency level?",
            "Should other areas be examined?",
            "Is sun protection important for this patient?",
            "What prevention measures should be taken?",
            "Is this lesion pigmented?",
            "Is this lesion raised?",
            "Is this lesion ulcerated?",
            "What is the size of this lesion?",
            "Are there multiple similar lesions?",
            "Is this lesion changing?",
            "Is this lesion symptomatic?",
            "Does this lesion bleed?",
            "Is this lesion itchy?",
            "What imaging might be helpful?",
            "Should the lesion be photographed for monitoring?",
            "What patient education is needed?",
            "Are there hereditary factors to consider?",
            "What lifestyle factors are relevant?",
            "Is this related to previous sun damage?",
            "Could this be related to immunosuppression?",
            "What differential diagnoses should be considered?",
            "Is this a primary or secondary lesion?",
            "What is the typical progression?",
            "Are there associated symptoms?",
            "What risk stratification is appropriate?",
            "Should family members be screened?",
            "What preventive measures are indicated?",
            "Is this work-related or occupational?",
            "What environmental factors are relevant?",
            "Should genetic counseling be considered?",
            "What comorbidities are relevant?",
            "Is medication history important?",
            "What is the disease burden?",
            "Are there quality of life considerations?",
            "What psychosocial factors are important?"
        ]
        
        self.comparison_questions = [
            "How does this compare to a normal mole?",
            "What features distinguish this from melanoma?",
            "How is this different from other skin cancers?",
            "What makes this lesion concerning?",
            "What features suggest benignity?",
            "How does this compare to seborrheic keratosis?",
            "What distinguishes this from a blood vessel lesion?",
            "How is this different from dermatofibroma?",
            "What features suggest malignancy?",
            "How does this compare to actinic keratosis?"
        ]
    
    def generate_diagnosis_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate diagnosis-specific questions."""
        questions = []
        diagnosis = sample.get('diagnosis', '').lower()
        
        if diagnosis in self.diagnosis_templates:
            templates = self.diagnosis_templates[diagnosis]
            
            # Add specific answers based on diagnosis
            answers = self._get_diagnosis_answers(diagnosis, sample)
            
            for i, template in enumerate(templates[:3]):  # Limit to 3 questions
                if i < len(answers):
                    questions.append({
                        'question': template,
                        'answer': answers[i],
                        'type': 'diagnosis_specific'
                    })
        
        return questions
    
    def generate_anatomical_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate anatomical site-specific questions."""
        questions = []
        site = sample.get('anatomical_site')
        
        if site and site in self.anatomical_questions:
            templates = self.anatomical_questions[site]
            
            for template in templates[:2]:  # Limit to 2 questions
                answer = self._get_anatomical_answer(template, site, sample)
                questions.append({
                    'question': template,
                    'answer': answer,
                    'type': 'anatomical'
                })
        
        return questions
    
    def generate_demographic_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate demographic-based questions."""
        questions = []
        
        # Age-based questions
        age = sample.get('age')
        if age is not None:
            age_templates = self.demographic_questions['age']
            template = random.choice(age_templates[:2])
            answer = self._get_age_answer(template, age)
            questions.append({
                'question': template,
                'answer': answer,
                'type': 'demographic_age'
            })
        
        # Sex-based questions
        sex = sample.get('sex')
        if sex and sex != 'unknown':
            sex_templates = self.demographic_questions['sex']
            template = random.choice(sex_templates[:2])
            answer = self._get_sex_answer(template, sex, sample.get('diagnosis'))
            questions.append({
                'question': template,
                'answer': answer,
                'type': 'demographic_sex'
            })
        
        return questions
    
    def generate_clinical_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate general clinical questions."""
        questions = []
        diagnosis = sample.get('diagnosis', '').lower()
        
        # Select relevant clinical questions based on diagnosis
        relevant_questions = random.sample(self.clinical_questions, min(3, len(self.clinical_questions)))
        
        for question in relevant_questions:
            answer = self._get_clinical_answer(question, sample)
            questions.append({
                'question': question,
                'answer': answer,
                'type': 'clinical'
            })
        
        return questions
    
    def generate_comparison_questions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate comparative questions."""
        questions = []
        diagnosis = sample.get('diagnosis', '').lower()
        
        # Select 1-2 comparison questions
        relevant_questions = random.sample(self.comparison_questions, min(2, len(self.comparison_questions)))
        
        for question in relevant_questions:
            answer = self._get_comparison_answer(question, sample)
            questions.append({
                'question': question,
                'answer': answer,
                'type': 'comparison'
            })
        
        return questions
    
    def _get_diagnosis_answers(self, diagnosis: str, sample: Dict[str, Any]) -> List[str]:
        """Get specific answers for diagnosis questions."""
        answers = []
        
        if diagnosis == 'melanoma':
            answers = [
                "Yes, this is a malignant lesion",
                "This is a melanoma, a type of skin cancer",
                "Yes, this is a melanoma",
                "Melanoma is the most concerning diagnosis",
                "Yes, immediate biopsy is indicated",
                "Yes, urgent medical attention is required",
                "The prognosis depends on staging and early detection",
                "Yes, melanoma can be life-threatening if not treated early",
                "Treatment typically involves surgical excision and may include adjuvant therapy"
            ]
        elif diagnosis == 'nevus':
            answers = [
                "Yes, this is a benign lesion",
                "Yes, this is a mole (nevus)",
                "This is a benign pigmented nevus",
                "Yes, this lesion is harmless",
                "No treatment is typically required for benign nevi",
                "Yes, nevi are very common skin lesions",
                "Regular monitoring for changes is recommended",
                "Most nevi remain stable over time"
            ]
        elif diagnosis == 'basal_cell_carcinoma':
            answers = [
                "This is a basal cell carcinoma",
                "Yes, this is a non-melanoma skin cancer",
                "Yes, this is malignant but rarely metastasizes",
                "Treatment typically involves surgical excision",
                "No, basal cell carcinoma rarely metastasizes",
                "Yes, it's the most common type of skin cancer",
                "UV exposure is the primary cause"
            ]
        elif diagnosis == 'squamous_cell_carcinoma':
            answers = [
                "This is a squamous cell carcinoma",
                "Yes, this is a malignant lesion",
                "Yes, this is a non-melanoma skin cancer",
                "There is some risk of metastasis, especially in high-risk locations",
                "Yes, UV exposure is a major risk factor",
                "Treatment involves surgical excision"
            ]
        elif diagnosis == 'actinic_keratosis':
            answers = [
                "Yes, this is a precancerous lesion",
                "There is a small risk of progression to squamous cell carcinoma",
                "Yes, chronic sun exposure is the primary cause",
                "Yes, treatment is recommended to prevent progression",
                "Yes, actinic keratoses are common in sun-exposed areas",
                "This represents early sun damage that should be treated"
            ]
        
        return answers
    
    def _get_anatomical_answer(self, question: str, site: str, sample: Dict[str, Any]) -> str:
        """Get anatomical site-specific answers."""
        site_names = {
            'head_neck': 'head and neck',
            'trunk': 'trunk/torso',
            'upper_extremity': 'upper extremity (arm/hand)',
            'lower_extremity': 'lower extremity (leg/foot)',
            'genitalia': 'genital area'
        }
        
        if 'where' in question.lower():
            return f"This lesion is located on the {site_names.get(site, site)}"
        elif 'sun-exposed' in question.lower():
            if site in ['head_neck', 'upper_extremity']:
                return "Yes, this is typically a sun-exposed area"
            else:
                return "This area may have variable sun exposure"
        elif 'common location' in question.lower():
            return f"Yes, the {site_names.get(site, site)} is a common location for skin lesions"
        elif 'body region' in question.lower():
            return site_names.get(site, site)
        else:
            return f"This affects the {site_names.get(site, site)}"
    
    def _get_age_answer(self, question: str, age: float) -> str:
        """Get age-specific answers."""
        if 'age group' in question.lower():
            if age < 18:
                return "This is a pediatric patient"
            elif age < 40:
                return "This is a young adult"
            elif age < 65:
                return "This is a middle-aged adult"
            else:
                return "This is an elderly patient"
        elif 'elderly' in question.lower():
            return "Yes" if age >= 65 else "No"
        elif 'pediatric' in question.lower():
            return "Yes" if age < 18 else "No"
        else:
            return f"The patient is {age} years old"
    
    def _get_sex_answer(self, question: str, sex: str, diagnosis: Optional[str]) -> str:
        """Get sex-specific answers."""
        if 'gender' in question.lower():
            return f"The patient is {sex}"
        elif 'predisposition' in question.lower():
            # Add some general knowledge about sex predispositions
            if diagnosis == 'melanoma':
                return "Melanoma rates are similar between males and females, though males have slightly higher rates"
            else:
                return "There may be slight variations between sexes for some conditions"
        else:
            return f"The patient is {sex}"
    
    def _get_clinical_answer(self, question: str, sample: Dict[str, Any]) -> str:
        """Get clinical answers based on diagnosis and context."""
        diagnosis = sample.get('diagnosis', '').lower()
        
        if 'biopsy' in question.lower():
            if diagnosis in ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']:
                return "Yes, biopsy is recommended for definitive diagnosis"
            elif diagnosis == 'actinic_keratosis':
                return "Biopsy may be considered if diagnosis is uncertain"
            else:
                return "Biopsy is typically not needed for clearly benign lesions"
        
        elif 'monitor' in question.lower():
            if diagnosis == 'nevus':
                return "Yes, regular monitoring for changes is recommended"
            else:
                return "Monitoring may be appropriate depending on the diagnosis"
        
        elif 'dermatologist' in question.lower():
            if diagnosis in ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']:
                return "Yes, dermatologist referral is indicated"
            else:
                return "Dermatologist consultation may be helpful"
        
        elif 'concerned' in question.lower():
            if diagnosis in ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']:
                return "Yes, this diagnosis requires attention and treatment"
            elif diagnosis == 'nevus':
                return "No, benign nevi are typically not concerning"
            else:
                return "The level of concern depends on the specific diagnosis"
        
        elif 'sun protection' in question.lower():
            return "Yes, sun protection is important for preventing skin cancer"
        
        else:
            # Generic clinical answer
            return "This depends on the specific clinical context and diagnosis"
    
    def _get_comparison_answer(self, question: str, sample: Dict[str, Any]) -> str:
        """Get comparative answers."""
        diagnosis = sample.get('diagnosis', '').lower()
        
        if 'normal mole' in question.lower():
            if diagnosis == 'melanoma':
                return "Melanoma often shows asymmetry, irregular borders, color variation, and diameter >6mm (ABCD criteria)"
            elif diagnosis == 'nevus':
                return "This appears consistent with a normal benign mole"
            else:
                return "This differs from a typical mole in its clinical appearance"
        
        elif 'melanoma' in question.lower():
            if diagnosis == 'melanoma':
                return "This lesion shows features consistent with melanoma"
            else:
                return f"This {diagnosis} typically lacks the irregular features of melanoma"
        
        elif 'concerning' in question.lower():
            if diagnosis in ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']:
                return "Features like asymmetry, irregular borders, and rapid growth make this concerning"
            else:
                return "This lesion shows reassuring features of a benign process"
        
        else:
            return "This lesion has specific features that help distinguish it from other skin conditions"
    
    def enrich_sample(self, sample: Dict[str, Any], max_new_questions: int = 10) -> Dict[str, Any]:
        """Enrich a single sample with additional VQA questions."""
        enriched_sample = sample.copy()
        
        # Get existing questions to avoid duplication
        existing_questions = {qa['question'].lower() for qa in sample.get('vqa_questions', [])}
        
        # Generate new questions
        new_questions = []
        
        # Add diagnosis-specific questions
        new_questions.extend(self.generate_diagnosis_questions(sample))
        
        # Add anatomical questions
        new_questions.extend(self.generate_anatomical_questions(sample))
        
        # Add demographic questions
        new_questions.extend(self.generate_demographic_questions(sample))
        
        # Add clinical questions
        new_questions.extend(self.generate_clinical_questions(sample))
        
        # Add comparison questions
        new_questions.extend(self.generate_comparison_questions(sample))
        
        # Filter out duplicates and limit number
        unique_new_questions = []
        for qa in new_questions:
            if qa['question'].lower() not in existing_questions and len(unique_new_questions) < max_new_questions:
                unique_new_questions.append(qa)
                existing_questions.add(qa['question'].lower())
        
        # Add new questions to existing ones
        enriched_sample['vqa_questions'] = sample.get('vqa_questions', []) + unique_new_questions
        
        return enriched_sample
    
    def enrich_dataset(self, input_file: Path, output_file: Path, max_new_questions: int = 10) -> Dict[str, Any]:
        """Enrich an entire dataset with additional VQA questions."""
        if not input_file.exists():
            return {'error': f'Input file not found: {input_file}'}
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
        
        enriched_data = []
        stats = {
            'total_samples': len(data),
            'total_original_questions': 0,
            'total_new_questions': 0,
            'avg_questions_per_sample_before': 0,
            'avg_questions_per_sample_after': 0
        }
        
        for sample in data:
            original_questions = len(sample.get('vqa_questions', []))
            stats['total_original_questions'] += original_questions
            
            enriched_sample = self.enrich_sample(sample, max_new_questions)
            new_questions = len(enriched_sample.get('vqa_questions', [])) - original_questions
            stats['total_new_questions'] += new_questions
            
            enriched_data.append(enriched_sample)
        
        # Calculate averages
        if stats['total_samples'] > 0:
            stats['avg_questions_per_sample_before'] = stats['total_original_questions'] / stats['total_samples']
            stats['avg_questions_per_sample_after'] = (stats['total_original_questions'] + stats['total_new_questions']) / stats['total_samples']
        
        # Save enriched data
        with open(output_file, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        
        return stats


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enrich dermatology VQA dataset with additional questions')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--max_questions', type=int, default=10, 
                       help='Maximum number of new questions per sample')
    
    args = parser.parse_args()
    
    enricher = VQAEnricher()
    stats = enricher.enrich_dataset(Path(args.input), Path(args.output), args.max_questions)
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
    else:
        print("✅ Dataset enrichment completed!")
        print(f"📊 Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Original questions: {stats['total_original_questions']}")
        print(f"  New questions added: {stats['total_new_questions']}")
        print(f"  Avg questions before: {stats['avg_questions_per_sample_before']:.1f}")
        print(f"  Avg questions after: {stats['avg_questions_per_sample_after']:.1f}")
        print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main() 