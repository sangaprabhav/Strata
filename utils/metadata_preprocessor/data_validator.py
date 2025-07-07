#!/usr/bin/env python3
"""
Data Validator for Dermatology VQA Dataset
Validates the quality and consistency of processed metadata.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter


class DataValidator:
    """Validates processed VQA dataset metadata."""
    
    def __init__(self):
        self.required_fields = [
            'dataset_name', 'sample_id', 'image_path', 'diagnosis',
            'age', 'sex', 'anatomical_site', 'metadata', 'vqa_questions'
        ]
        self.valid_diagnoses = [
            'melanoma', 'nevus', 'benign', 'benign_keratosis', 
            'basal_cell_carcinoma', 'actinic_keratosis', 'dermatofibroma',
            'vascular_lesion', 'squamous_cell_carcinoma', 'unknown'
        ]
        self.valid_sexes = ['male', 'female', 'unknown', None]
        self.valid_anatomical_sites = [
            'head_neck', 'trunk', 'upper_extremity', 'lower_extremity',
            'genitalia', 'unknown', None
        ]
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single processed JSON file."""
        if not file_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
        
        if not isinstance(data, list):
            return {'error': 'Data should be a list of samples'}
        
        validation_results = {
            'file_path': str(file_path),
            'total_samples': len(data),
            'valid_samples': 0,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        for i, sample in enumerate(data):
            sample_errors, sample_warnings = self.validate_sample(sample, i)
            validation_results['errors'].extend(sample_errors)
            validation_results['warnings'].extend(sample_warnings)
            
            if not sample_errors:
                validation_results['valid_samples'] += 1
        
        # Generate statistics
        validation_results['statistics'] = self.generate_statistics(data)
        
        return validation_results
    
    def validate_sample(self, sample: Dict[str, Any], index: int) -> Tuple[List[str], List[str]]:
        """Validate a single sample."""
        errors = []
        warnings = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in sample:
                errors.append(f"Sample {index}: Missing required field '{field}'")
        
        # Validate specific fields
        if 'diagnosis' in sample:
            if sample['diagnosis'] not in self.valid_diagnoses:
                warnings.append(f"Sample {index}: Unusual diagnosis '{sample['diagnosis']}'")
        
        if 'sex' in sample:
            if sample['sex'] not in self.valid_sexes:
                warnings.append(f"Sample {index}: Unusual sex value '{sample['sex']}'")
        
        if 'anatomical_site' in sample:
            if sample['anatomical_site'] not in self.valid_anatomical_sites:
                warnings.append(f"Sample {index}: Unusual anatomical site '{sample['anatomical_site']}'")
        
        if 'age' in sample and sample['age'] is not None:
            try:
                age = float(sample['age'])
                if age < 0 or age > 150:
                    warnings.append(f"Sample {index}: Unusual age value {age}")
            except (ValueError, TypeError):
                errors.append(f"Sample {index}: Invalid age format '{sample['age']}'")
        
        # Validate VQA questions
        if 'vqa_questions' in sample:
            if not isinstance(sample['vqa_questions'], list):
                errors.append(f"Sample {index}: vqa_questions should be a list")
            else:
                for j, qa in enumerate(sample['vqa_questions']):
                    if not isinstance(qa, dict):
                        errors.append(f"Sample {index}, QA {j}: Should be a dictionary")
                        continue
                    
                    if 'question' not in qa or 'answer' not in qa:
                        errors.append(f"Sample {index}, QA {j}: Missing 'question' or 'answer'")
                    
                    if not qa.get('question', '').strip():
                        warnings.append(f"Sample {index}, QA {j}: Empty question")
                    
                    if not qa.get('answer', '').strip():
                        warnings.append(f"Sample {index}, QA {j}: Empty answer")
        
        # Validate image path
        if 'image_path' in sample:
            if not sample['image_path'] or not sample['image_path'].strip():
                errors.append(f"Sample {index}: Empty image path")
        
        return errors, warnings
    
    def generate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics for the dataset."""
        stats = {}
        
        # Dataset distribution
        datasets = [sample.get('dataset_name', 'unknown') for sample in data]
        stats['dataset_distribution'] = dict(Counter(datasets))
        
        # Diagnosis distribution
        diagnoses = [sample.get('diagnosis', 'unknown') for sample in data]
        stats['diagnosis_distribution'] = dict(Counter(diagnoses))
        
        # Sex distribution
        sexes = [sample.get('sex', 'unknown') for sample in data if sample.get('sex')]
        stats['sex_distribution'] = dict(Counter(sexes))
        
        # Anatomical site distribution
        sites = [sample.get('anatomical_site', 'unknown') for sample in data if sample.get('anatomical_site')]
        stats['anatomical_site_distribution'] = dict(Counter(sites))
        
        # Age statistics
        ages = [sample.get('age') for sample in data if sample.get('age') is not None]
        if ages:
            ages = [float(age) for age in ages if isinstance(age, (int, float)) or str(age).replace('.', '').isdigit()]
            if ages:
                stats['age_statistics'] = {
                    'count': len(ages),
                    'mean': sum(ages) / len(ages),
                    'min': min(ages),
                    'max': max(ages),
                    'median': sorted(ages)[len(ages)//2]
                }
        
        # VQA question statistics
        total_questions = sum(len(sample.get('vqa_questions', [])) for sample in data)
        stats['vqa_statistics'] = {
            'total_questions': total_questions,
            'avg_questions_per_sample': total_questions / len(data) if data else 0
        }
        
        return stats
    
    def validate_directory(self, directory: Path) -> Dict[str, Any]:
        """Validate all processed JSON files in a directory."""
        if not directory.exists():
            return {'error': f'Directory not found: {directory}'}
        
        json_files = list(directory.glob('*_processed.json'))
        if not json_files:
            return {'error': 'No processed JSON files found'}
        
        overall_results = {
            'directory': str(directory),
            'total_files': len(json_files),
            'file_results': {},
            'summary': {
                'total_samples': 0,
                'total_valid_samples': 0,
                'total_errors': 0,
                'total_warnings': 0
            }
        }
        
        for json_file in json_files:
            file_results = self.validate_file(json_file)
            dataset_name = json_file.stem.replace('_processed', '')
            overall_results['file_results'][dataset_name] = file_results
            
            if 'error' not in file_results:
                overall_results['summary']['total_samples'] += file_results['total_samples']
                overall_results['summary']['total_valid_samples'] += file_results['valid_samples']
                overall_results['summary']['total_errors'] += len(file_results['errors'])
                overall_results['summary']['total_warnings'] += len(file_results['warnings'])
        
        return overall_results
    
    def generate_report(self, validation_results: Dict[str, Any], output_file: Path = None) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        
        if 'error' in validation_results:
            report_lines.append(f"❌ VALIDATION FAILED: {validation_results['error']}")
            return '\n'.join(report_lines)
        
        # Overall summary
        if 'summary' in validation_results:  # Directory validation
            summary = validation_results['summary']
            report_lines.extend([
                "📊 DATASET VALIDATION REPORT",
                "=" * 50,
                f"Directory: {validation_results['directory']}",
                f"Total files: {validation_results['total_files']}",
                f"Total samples: {summary['total_samples']}",
                f"Valid samples: {summary['total_valid_samples']}",
                f"Success rate: {summary['total_valid_samples']/summary['total_samples']*100:.1f}%" if summary['total_samples'] > 0 else "Success rate: 0%",
                f"Total errors: {summary['total_errors']}",
                f"Total warnings: {summary['total_warnings']}",
                ""
            ])
            
            # File-by-file results
            for dataset_name, file_result in validation_results['file_results'].items():
                if 'error' in file_result:
                    report_lines.append(f"❌ {dataset_name}: {file_result['error']}")
                else:
                    success_rate = file_result['valid_samples'] / file_result['total_samples'] * 100 if file_result['total_samples'] > 0 else 0
                    status = "✅" if len(file_result['errors']) == 0 else "⚠️"
                    report_lines.append(f"{status} {dataset_name}: {file_result['valid_samples']}/{file_result['total_samples']} valid ({success_rate:.1f}%)")
                    
                    if file_result['errors']:
                        report_lines.append(f"   Errors: {len(file_result['errors'])}")
                    if file_result['warnings']:
                        report_lines.append(f"   Warnings: {len(file_result['warnings'])}")
        
        else:  # Single file validation
            total_samples = validation_results['total_samples']
            valid_samples = validation_results['valid_samples']
            success_rate = valid_samples / total_samples * 100 if total_samples > 0 else 0
            
            report_lines.extend([
                "📊 FILE VALIDATION REPORT",
                "=" * 50,
                f"File: {validation_results['file_path']}",
                f"Total samples: {total_samples}",
                f"Valid samples: {valid_samples}",
                f"Success rate: {success_rate:.1f}%",
                f"Errors: {len(validation_results['errors'])}",
                f"Warnings: {len(validation_results['warnings'])}",
                ""
            ])
            
            # Show statistics if available
            if 'statistics' in validation_results:
                stats = validation_results['statistics']
                report_lines.append("📈 STATISTICS")
                report_lines.append("-" * 20)
                
                for stat_name, stat_data in stats.items():
                    if isinstance(stat_data, dict):
                        report_lines.append(f"{stat_name.replace('_', ' ').title()}:")
                        for key, value in stat_data.items():
                            report_lines.append(f"  {key}: {value}")
                    else:
                        report_lines.append(f"{stat_name.replace('_', ' ').title()}: {stat_data}")
                
                report_lines.append("")
        
        # Show errors and warnings
        if 'file_results' in validation_results:
            # Collect all errors and warnings from all files
            all_errors = []
            all_warnings = []
            for file_result in validation_results['file_results'].values():
                if 'errors' in file_result:
                    all_errors.extend(file_result['errors'])
                if 'warnings' in file_result:
                    all_warnings.extend(file_result['warnings'])
        else:
            all_errors = validation_results.get('errors', [])
            all_warnings = validation_results.get('warnings', [])
        
        if all_errors:
            report_lines.extend([
                "❌ ERRORS",
                "-" * 20
            ])
            for error in all_errors[:10]:  # Show first 10 errors
                report_lines.append(f"  • {error}")
            if len(all_errors) > 10:
                report_lines.append(f"  ... and {len(all_errors) - 10} more errors")
            report_lines.append("")
        
        if all_warnings:
            report_lines.extend([
                "⚠️  WARNINGS",
                "-" * 20
            ])
            for warning in all_warnings[:10]:  # Show first 10 warnings
                report_lines.append(f"  • {warning}")
            if len(all_warnings) > 10:
                report_lines.append(f"  ... and {len(all_warnings) - 10} more warnings")
            report_lines.append("")
        
        report = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dermatology VQA dataset metadata')
    parser.add_argument('--file', type=str, help='Validate a specific JSON file')
    parser.add_argument('--directory', type=str, default='./processed_metadata',
                       help='Validate all files in directory')
    parser.add_argument('--output', type=str, help='Output report file')
    
    args = parser.parse_args()
    
    validator = DataValidator()
    
    if args.file:
        # Validate single file
        results = validator.validate_file(Path(args.file))
    else:
        # Validate directory
        results = validator.validate_directory(Path(args.directory))
    
    # Generate and display report
    output_file = Path(args.output) if args.output else None
    report = validator.generate_report(results, output_file)
    print(report)


if __name__ == "__main__":
    main() 