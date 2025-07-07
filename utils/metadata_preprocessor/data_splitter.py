#!/usr/bin/env python3
"""
Data Splitter for Dermatology VQA Dataset
Creates train/validation/test splits with stratification support.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter


class DataSplitter:
    """Splits VQA dataset into train/validation/test sets."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def stratified_split(self, data: List[Dict[str, Any]], 
                        train_ratio: float = 0.7, 
                        val_ratio: float = 0.15, 
                        test_ratio: float = 0.15,
                        stratify_by: str = 'diagnosis') -> Tuple[List, List, List]:
        """Split data with stratification to maintain class balance."""
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Group samples by stratification key
        groups = {}
        for sample in data:
            key = sample.get(stratify_by, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        train_data, val_data, test_data = [], [], []
        
        for key, samples in groups.items():
            # Shuffle samples within each group
            random.shuffle(samples)
            
            n_samples = len(samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # Ensure at least one sample per split if possible
            if n_samples >= 3:
                if n_train == 0:
                    n_train = 1
                if n_val == 0:
                    n_val = 1
                
                # Adjust if needed
                n_test = n_samples - n_train - n_val
                if n_test < 0:
                    n_test = 0
                    n_val = n_samples - n_train
            elif n_samples == 2:
                n_train, n_val, n_test = 1, 1, 0
            elif n_samples == 1:
                n_train, n_val, n_test = 1, 0, 0
            else:
                continue
            
            # Split the samples
            train_data.extend(samples[:n_train])
            val_data.extend(samples[n_train:n_train + n_val])
            test_data.extend(samples[n_train + n_val:])
        
        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def random_split(self, data: List[Dict[str, Any]], 
                    train_ratio: float = 0.7, 
                    val_ratio: float = 0.15, 
                    test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """Split data randomly without stratification."""
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle data
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n_samples = len(data_copy)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data = data_copy[:n_train]
        val_data = data_copy[n_train:n_train + n_val]
        test_data = data_copy[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def dataset_aware_split(self, data: List[Dict[str, Any]], 
                           train_ratio: float = 0.7, 
                           val_ratio: float = 0.15, 
                           test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """Split data ensuring samples from same dataset are not split across train/val/test."""
        
        # Group by dataset
        dataset_groups = {}
        for sample in data:
            dataset_name = sample.get('dataset_name', 'unknown')
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(sample)
        
        # Determine dataset assignment to splits
        datasets = list(dataset_groups.keys())
        random.shuffle(datasets)
        
        n_datasets = len(datasets)
        n_train_datasets = max(1, int(n_datasets * train_ratio))
        n_val_datasets = max(1, int(n_datasets * val_ratio)) if n_datasets > 1 else 0
        
        train_datasets = datasets[:n_train_datasets]
        val_datasets = datasets[n_train_datasets:n_train_datasets + n_val_datasets]
        test_datasets = datasets[n_train_datasets + n_val_datasets:]
        
        # Collect samples
        train_data = []
        val_data = []
        test_data = []
        
        for dataset in train_datasets:
            train_data.extend(dataset_groups[dataset])
        
        for dataset in val_datasets:
            val_data.extend(dataset_groups[dataset])
        
        for dataset in test_datasets:
            test_data.extend(dataset_groups[dataset])
        
        return train_data, val_data, test_data
    
    def split_dataset(self, input_file: Path, 
                     output_dir: Path,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     split_method: str = 'stratified',
                     stratify_by: str = 'diagnosis') -> Dict[str, Any]:
        """Split a dataset and save the splits."""
        
        if not input_file.exists():
            return {'error': f'Input file not found: {input_file}'}
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split the data
        if split_method == 'stratified':
            train_data, val_data, test_data = self.stratified_split(
                data, train_ratio, val_ratio, test_ratio, stratify_by)
        elif split_method == 'dataset_aware':
            train_data, val_data, test_data = self.dataset_aware_split(
                data, train_ratio, val_ratio, test_ratio)
        else:  # random
            train_data, val_data, test_data = self.random_split(
                data, train_ratio, val_ratio, test_ratio)
        
        # Save splits
        base_name = input_file.stem
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        split_info = {
            'input_file': str(input_file),
            'total_samples': len(data),
            'split_method': split_method,
            'random_seed': self.random_seed,
            'splits': {}
        }
        
        for split_name, split_data in splits.items():
            if split_data:  # Only save non-empty splits
                output_file = output_dir / f"{base_name}_{split_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(split_data, f, indent=2)
                
                # Collect split statistics
                split_stats = self._get_split_statistics(split_data)
                split_info['splits'][split_name] = {
                    'file': str(output_file),
                    'samples': len(split_data),
                    'percentage': len(split_data) / len(data) * 100,
                    'statistics': split_stats
                }
        
        # Save split information
        info_file = output_dir / f"{base_name}_split_info.json"
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return split_info
    
    def _get_split_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics for a data split."""
        stats = {}
        
        # Dataset distribution
        datasets = [sample.get('dataset_name', 'unknown') for sample in data]
        stats['dataset_distribution'] = dict(Counter(datasets))
        
        # Diagnosis distribution
        diagnoses = [sample.get('diagnosis', 'unknown') for sample in data]
        stats['diagnosis_distribution'] = dict(Counter(diagnoses))
        
        # Sex distribution
        sexes = [sample.get('sex', 'unknown') for sample in data if sample.get('sex')]
        if sexes:
            stats['sex_distribution'] = dict(Counter(sexes))
        
        # Anatomical site distribution
        sites = [sample.get('anatomical_site', 'unknown') for sample in data if sample.get('anatomical_site')]
        if sites:
            stats['anatomical_site_distribution'] = dict(Counter(sites))
        
        # VQA statistics
        total_questions = sum(len(sample.get('vqa_questions', [])) for sample in data)
        stats['vqa_statistics'] = {
            'total_questions': total_questions,
            'avg_questions_per_sample': total_questions / len(data) if data else 0
        }
        
        return stats
    
    def analyze_splits(self, split_info_file: Path) -> str:
        """Analyze the quality of data splits."""
        if not split_info_file.exists():
            return f"Split info file not found: {split_info_file}"
        
        with open(split_info_file, 'r') as f:
            split_info = json.load(f)
        
        report_lines = [
            "📊 SPLIT ANALYSIS REPORT",
            "=" * 50,
            f"Input file: {split_info['input_file']}",
            f"Total samples: {split_info['total_samples']}",
            f"Split method: {split_info['split_method']}",
            f"Random seed: {split_info['random_seed']}",
            ""
        ]
        
        # Split sizes
        report_lines.append("📈 Split Sizes:")
        for split_name, split_data in split_info['splits'].items():
            report_lines.append(f"  {split_name}: {split_data['samples']} samples ({split_data['percentage']:.1f}%)")
        
        report_lines.append("")
        
        # Diagnosis distribution across splits
        report_lines.append("🏥 Diagnosis Distribution:")
        all_diagnoses = set()
        for split_data in split_info['splits'].values():
            all_diagnoses.update(split_data['statistics']['diagnosis_distribution'].keys())
        
        for diagnosis in sorted(all_diagnoses):
            report_lines.append(f"  {diagnosis}:")
            for split_name, split_data in split_info['splits'].items():
                count = split_data['statistics']['diagnosis_distribution'].get(diagnosis, 0)
                percentage = count / split_data['samples'] * 100 if split_data['samples'] > 0 else 0
                report_lines.append(f"    {split_name}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Dataset distribution across splits
        report_lines.append("📚 Dataset Distribution:")
        all_datasets = set()
        for split_data in split_info['splits'].values():
            all_datasets.update(split_data['statistics']['dataset_distribution'].keys())
        
        for dataset in sorted(all_datasets):
            report_lines.append(f"  {dataset}:")
            for split_name, split_data in split_info['splits'].items():
                count = split_data['statistics']['dataset_distribution'].get(dataset, 0)
                percentage = count / split_data['samples'] * 100 if split_data['samples'] > 0 else 0
                report_lines.append(f"    {split_name}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # VQA statistics
        report_lines.append("❓ VQA Statistics:")
        for split_name, split_data in split_info['splits'].items():
            vqa_stats = split_data['statistics']['vqa_statistics']
            report_lines.append(f"  {split_name}:")
            report_lines.append(f"    Total questions: {vqa_stats['total_questions']}")
            report_lines.append(f"    Avg per sample: {vqa_stats['avg_questions_per_sample']:.1f}")
        
        return '\n'.join(report_lines)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dermatology VQA dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output_dir', type=str, default='./splits', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--method', type=str, choices=['random', 'stratified', 'dataset_aware'], 
                       default='stratified', help='Splitting method')
    parser.add_argument('--stratify_by', type=str, default='diagnosis', 
                       help='Field to stratify by (for stratified method)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--analyze', type=str, help='Analyze existing split info file')
    
    args = parser.parse_args()
    
    splitter = DataSplitter(args.seed)
    
    if args.analyze:
        # Analyze existing splits
        report = splitter.analyze_splits(Path(args.analyze))
        print(report)
    else:
        # Create new splits
        split_info = splitter.split_dataset(
            input_file=Path(args.input),
            output_dir=Path(args.output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_method=args.method,
            stratify_by=args.stratify_by
        )
        
        if 'error' in split_info:
            print(f"❌ Error: {split_info['error']}")
        else:
            print("✅ Dataset splitting completed!")
            print(f"📊 Statistics:")
            print(f"  Total samples: {split_info['total_samples']}")
            print(f"  Split method: {split_info['split_method']}")
            
            for split_name, split_data in split_info['splits'].items():
                print(f"  {split_name}: {split_data['samples']} samples ({split_data['percentage']:.1f}%)")
            
            print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 