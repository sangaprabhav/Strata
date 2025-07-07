#!/usr/bin/env python3
"""
Main script for preprocessing metadata from various dermatology datasets
for Visual Question Answering (VQA) dataset creation.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the current directory to sys.path to import processors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bcn20k_processor import BCN20KProcessor
from ddi_processor import DDIProcessor
from ddi2_processor import DDI2Processor
from derm12345_processor import Derm12345Processor
from ham10k_processor import HAM10KProcessor
from hiba_processor import HIBAProcessor
from isic2020_processor import ISIC2020Processor
from mra_midas_processor import MRAMIDASProcessor
from mskcc_processor import MSKCCProcessor
from pad_ufes20_processor import PADUfes20Processor
from patch16_processor import Patch16Processor
from scin_processor import SCINProcessor


class MetadataPreprocessor:
    """Main class for preprocessing metadata from various dermatology datasets."""
    
    def __init__(self, datasets_dir="../datasets", output_dir="./processed_metadata"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all processors
        self.processors = {
            'bcn20k': BCN20KProcessor(),
            'ddi': DDIProcessor(),
            'ddi-2': DDI2Processor(),
            'derm12345': Derm12345Processor(),
            'ham10k': HAM10KProcessor(),
            'hiba': HIBAProcessor(),
            'isic2020': ISIC2020Processor(),
            'mra-midas': MRAMIDASProcessor(),
            'mskcc': MSKCCProcessor(),
            'pad-ufes20': PADUfes20Processor(),
            'patch16': Patch16Processor(),
            'scin': SCINProcessor()
        }
    
    def process_dataset(self, dataset_name):
        """Process a single dataset."""
        if dataset_name not in self.processors:
            print(f"Unknown dataset: {dataset_name}")
            return False
        
        processor = self.processors[dataset_name]
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"Dataset directory not found: {dataset_path}")
            return False
        
        try:
            print(f"Processing {dataset_name}...")
            processed_data = processor.process(dataset_path)
            
            # Save processed data
            output_file = self.output_dir / f"{dataset_name}_processed.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"Processed {len(processed_data)} samples from {dataset_name}")
            print(f"Output saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            return False
    
    def process_all_datasets(self):
        """Process all available datasets."""
        results = {}
        
        for dataset_name in self.processors.keys():
            results[dataset_name] = self.process_dataset(dataset_name)
        
        # Generate summary
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        print(f"\nSummary:")
        print(f"Successfully processed: {len(successful)} datasets")
        print(f"Failed to process: {len(failed)} datasets")
        
        if successful:
            print(f"Successful: {', '.join(successful)}")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        
        return results
    
    def combine_all_datasets(self):
        """Combine all processed datasets into a single JSON file."""
        combined_data = []
        
        for dataset_name in self.processors.keys():
            output_file = self.output_dir / f"{dataset_name}_processed.json"
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    combined_data.extend(data)
        
        # Save combined data
        combined_file = self.output_dir / "combined_metadata.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"Combined {len(combined_data)} samples from all datasets")
        print(f"Combined output saved to: {combined_file}")
        
        return combined_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess dermatology dataset metadata for VQA')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets_dir', type=str, default='../datasets', 
                       help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='./processed_metadata',
                       help='Directory to save processed metadata')
    parser.add_argument('--combine', action='store_true', 
                       help='Combine all processed datasets into a single file')
    
    args = parser.parse_args()
    
    preprocessor = MetadataPreprocessor(args.datasets_dir, args.output_dir)
    
    if args.dataset:
        # Process specific dataset
        preprocessor.process_dataset(args.dataset)
    else:
        # Process all datasets
        preprocessor.process_all_datasets()
    
    if args.combine:
        # Combine all datasets
        preprocessor.combine_all_datasets()


if __name__ == "__main__":
    main() 