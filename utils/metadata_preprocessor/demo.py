#!/usr/bin/env python3
"""
Demo Script for Dermatology VQA Dataset Preprocessing Pipeline
Demonstrates the complete workflow from raw metadata to VQA-ready splits.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from main import MetadataProcessor
from data_validator import DataValidator
from vqa_enricher import VQAEnricher
from data_splitter import DataSplitter


class VQAPipeline:
    """Complete VQA preprocessing pipeline."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.processor = MetadataProcessor()
        self.validator = DataValidator()
        self.enricher = VQAEnricher()
        self.splitter = DataSplitter()
        
        # Create output directories
        self.output_dir = self.base_dir / "vqa_output"
        self.processed_dir = self.output_dir / "processed"
        self.enriched_dir = self.output_dir / "enriched"
        self.splits_dir = self.output_dir / "splits"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.processed_dir, self.enriched_dir, self.splits_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_complete_pipeline(self, dataset_name: str = None) -> Dict[str, Any]:
        """Run the complete VQA preprocessing pipeline."""
        
        print("🚀 Starting VQA Preprocessing Pipeline")
        print("=" * 50)
        
        results = {
            'steps_completed': [],
            'errors': [],
            'outputs': {}
        }
        
        try:
            # Step 1: Process raw metadata
            print("\n📊 Step 1: Processing Raw Metadata")
            print("-" * 30)
            
            if dataset_name:
                # Process single dataset
                process_results = self.processor.process_single_dataset(dataset_name)
                if process_results.get('success'):
                    print(f"✅ Successfully processed {dataset_name}")
                    results['steps_completed'].append('single_dataset_processing')
                else:
                    print(f"❌ Failed to process {dataset_name}: {process_results.get('error')}")
                    results['errors'].append(f"Processing {dataset_name}: {process_results.get('error')}")
                    return results
            else:
                # Process all datasets
                process_results = self.processor.process_all_datasets()
                print(f"✅ Processed {len(process_results)} datasets")
                results['steps_completed'].append('all_datasets_processing')
            
            # Step 2: Validate processed data
            print("\n🔍 Step 2: Validating Processed Data")
            print("-" * 30)
            
            validation_results = self.validator.validate_directory(
                self.base_dir / "processed_metadata"
            )
            
            if 'error' not in validation_results:
                summary = validation_results['summary']
                success_rate = summary['total_valid_samples'] / summary['total_samples'] * 100 if summary['total_samples'] > 0 else 0
                print(f"✅ Validation completed: {success_rate:.1f}% success rate")
                print(f"   Total samples: {summary['total_samples']}")
                print(f"   Valid samples: {summary['total_valid_samples']}")
                print(f"   Errors: {summary['total_errors']}")
                print(f"   Warnings: {summary['total_warnings']}")
                
                # Save validation report
                report_file = self.reports_dir / "validation_report.txt"
                validation_report = self.validator.generate_report(validation_results, report_file)
                results['outputs']['validation_report'] = str(report_file)
                results['steps_completed'].append('validation')
            else:
                print(f"❌ Validation failed: {validation_results['error']}")
                results['errors'].append(f"Validation: {validation_results['error']}")
                return results
            
            # Step 3: Enrich with additional VQA questions
            print("\n❓ Step 3: Enriching VQA Questions")
            print("-" * 30)
            
            processed_files = list((self.base_dir / "processed_metadata").glob("*_processed.json"))
            if not processed_files:
                print("❌ No processed files found for enrichment")
                results['errors'].append("No processed files found for enrichment")
                return results
            
            # For demo, enrich the first file or specific dataset
            target_file = processed_files[0]
            if dataset_name:
                target_files = [f for f in processed_files if dataset_name in f.name]
                if target_files:
                    target_file = target_files[0]
            
            enriched_file = self.enriched_dir / f"{target_file.stem}_enriched.json"
            enrich_stats = self.enricher.enrich_dataset(target_file, enriched_file, max_new_questions=8)
            
            if 'error' not in enrich_stats:
                print(f"✅ Enrichment completed for {target_file.name}")
                print(f"   Original questions: {enrich_stats['total_original_questions']}")
                print(f"   New questions: {enrich_stats['total_new_questions']}")
                print(f"   Avg questions per sample: {enrich_stats['avg_questions_per_sample_after']:.1f}")
                results['outputs']['enriched_file'] = str(enriched_file)
                results['steps_completed'].append('enrichment')
            else:
                print(f"❌ Enrichment failed: {enrich_stats['error']}")
                results['errors'].append(f"Enrichment: {enrich_stats['error']}")
                return results
            
            # Step 4: Create train/validation/test splits
            print("\n🔄 Step 4: Creating Data Splits")
            print("-" * 30)
            
            split_info = self.splitter.split_dataset(
                input_file=enriched_file,
                output_dir=self.splits_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                split_method='stratified',
                stratify_by='diagnosis'
            )
            
            if 'error' not in split_info:
                print(f"✅ Data splitting completed")
                print(f"   Total samples: {split_info['total_samples']}")
                for split_name, split_data in split_info['splits'].items():
                    print(f"   {split_name}: {split_data['samples']} samples ({split_data['percentage']:.1f}%)")
                
                # Generate split analysis report
                split_info_file = self.splits_dir / f"{enriched_file.stem}_split_info.json"
                split_report = self.splitter.analyze_splits(split_info_file)
                split_report_file = self.reports_dir / "split_analysis_report.txt"
                with open(split_report_file, 'w') as f:
                    f.write(split_report)
                
                results['outputs']['split_info'] = str(split_info_file)
                results['outputs']['split_report'] = str(split_report_file)
                results['steps_completed'].append('splitting')
            else:
                print(f"❌ Data splitting failed: {split_info['error']}")
                results['errors'].append(f"Splitting: {split_info['error']}")
                return results
            
            # Step 5: Final validation of splits
            print("\n🔍 Step 5: Final Validation")
            print("-" * 30)
            
            split_files = list(self.splits_dir.glob("*_train.json"))
            if split_files:
                final_validation = self.validator.validate_file(split_files[0])
                if 'error' not in final_validation:
                    print(f"✅ Final validation passed")
                    print(f"   Training set: {final_validation['valid_samples']}/{final_validation['total_samples']} valid samples")
                    results['steps_completed'].append('final_validation')
                else:
                    print(f"⚠️ Final validation issues: {final_validation['error']}")
                    results['errors'].append(f"Final validation: {final_validation['error']}")
            
            # Step 6: Generate summary
            print("\n📋 Step 6: Generating Summary")
            print("-" * 30)
            
            summary = self.generate_pipeline_summary(results)
            summary_file = self.reports_dir / "pipeline_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            results['outputs']['summary'] = str(summary_file)
            results['steps_completed'].append('summary')
            
            print(f"✅ Pipeline completed successfully!")
            print(f"📁 All outputs saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            results['errors'].append(f"Pipeline error: {e}")
        
        return results
    
    def generate_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the pipeline execution."""
        
        summary_lines = [
            "🎯 VQA PREPROCESSING PIPELINE SUMMARY",
            "=" * 50,
            f"Execution timestamp: {self.get_timestamp()}",
            f"Base directory: {self.base_dir}",
            f"Output directory: {self.output_dir}",
            "",
            "📊 Steps Completed:",
        ]
        
        step_descriptions = {
            'single_dataset_processing': '✅ Single dataset processing',
            'all_datasets_processing': '✅ All datasets processing',
            'validation': '✅ Data validation',
            'enrichment': '✅ VQA question enrichment',
            'splitting': '✅ Train/validation/test splitting',
            'final_validation': '✅ Final validation',
            'summary': '✅ Summary generation'
        }
        
        for step in results['steps_completed']:
            summary_lines.append(f"  {step_descriptions.get(step, step)}")
        
        if results['errors']:
            summary_lines.extend([
                "",
                "❌ Errors Encountered:",
            ])
            for error in results['errors']:
                summary_lines.append(f"  • {error}")
        
        if results['outputs']:
            summary_lines.extend([
                "",
                "📁 Generated Files:",
            ])
            for output_name, output_path in results['outputs'].items():
                summary_lines.append(f"  {output_name}: {output_path}")
        
        summary_lines.extend([
            "",
            "🚀 Next Steps:",
            "  1. Review validation and split analysis reports",
            "  2. Inspect sample data to ensure quality",
            "  3. Use the train/val/test splits for VQA model training",
            "  4. Consider additional data augmentation if needed",
            "",
            "📖 For more information, see the README.md file",
        ])
        
        return '\n'.join(summary_lines)
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def quick_demo(self, dataset_name: str = 'ham10k') -> None:
        """Run a quick demo with a specific dataset."""
        
        print(f"🎬 Quick Demo: Processing {dataset_name.upper()} dataset")
        print("=" * 50)
        
        # Check if dataset exists
        dataset_path = self.base_dir / "datasets" / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_path}")
            return
        
        # Run the pipeline
        results = self.run_complete_pipeline(dataset_name)
        
        print("\n🎯 Demo Results:")
        print(f"  Steps completed: {len(results['steps_completed'])}")
        print(f"  Errors: {len(results['errors'])}")
        print(f"  Output files: {len(results['outputs'])}")
        
        if results['outputs']:
            print(f"  Check outputs in: {self.output_dir}")


def main():
    """Main function for demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo VQA preprocessing pipeline')
    parser.add_argument('--dataset', type=str, 
                       help='Specific dataset to process (e.g., ham10k, ddi, isic2020)')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory containing datasets')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with ham10k dataset')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VQAPipeline(Path(args.base_dir))
    
    if args.quick:
        # Quick demo
        pipeline.quick_demo()
    else:
        # Full pipeline
        results = pipeline.run_complete_pipeline(args.dataset)
        
        print("\n📊 Final Results:")
        print(f"  Success: {len(results['errors']) == 0}")
        print(f"  Steps completed: {len(results['steps_completed'])}")
        print(f"  Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  • {error}")


if __name__ == "__main__":
    main() 