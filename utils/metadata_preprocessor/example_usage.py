#!/usr/bin/env python3
"""
Example usage of the metadata preprocessor for dermatology VQA dataset creation.
"""

import json
from pathlib import Path
from main import MetadataPreprocessor

def example_single_dataset():
    """Example of processing a single dataset."""
    print("=== Processing Single Dataset Example ===")
    
    # Initialize the preprocessor
    preprocessor = MetadataPreprocessor(
        datasets_dir="../../datasets",
        output_dir="./example_output"
    )
    
    # Process HAM10K dataset
    success = preprocessor.process_dataset('ham10k')
    
    if success:
        print("✓ HAM10K dataset processed successfully")
        
        # Load and display sample results
        output_file = Path("./example_output/ham10k_processed.json")
        if output_file.exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"✓ Processed {len(data)} samples")
                
                # Display first sample
                if data:
                    print("\nSample output:")
                    print(json.dumps(data[0], indent=2))
    else:
        print("✗ Failed to process HAM10K dataset")

def example_all_datasets():
    """Example of processing all datasets."""
    print("\n=== Processing All Datasets Example ===")
    
    # Initialize the preprocessor
    preprocessor = MetadataPreprocessor(
        datasets_dir="../../datasets",
        output_dir="./example_output"
    )
    
    # Process all datasets
    results = preprocessor.process_all_datasets()
    
    # Display results
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"✓ Successfully processed: {len(successful)} datasets")
    print(f"✗ Failed to process: {len(failed)} datasets")
    
    if successful:
        print(f"Successful datasets: {', '.join(successful)}")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")

def example_combine_datasets():
    """Example of combining all processed datasets."""
    print("\n=== Combining Datasets Example ===")
    
    # Initialize the preprocessor
    preprocessor = MetadataPreprocessor(
        datasets_dir="../../datasets",
        output_dir="./example_output"
    )
    
    # Combine all processed datasets
    combined_data = preprocessor.combine_all_datasets()
    
    print(f"✓ Combined {len(combined_data)} total samples")
    
    # Display statistics
    if combined_data:
        datasets = {}
        diagnoses = {}
        
        for sample in combined_data:
            dataset_name = sample.get('dataset_name', 'unknown')
            diagnosis = sample.get('diagnosis', 'unknown')
            
            datasets[dataset_name] = datasets.get(dataset_name, 0) + 1
            diagnoses[diagnosis] = diagnoses.get(diagnosis, 0) + 1
        
        print("\nDataset distribution:")
        for dataset, count in sorted(datasets.items()):
            print(f"  {dataset}: {count} samples")
        
        print("\nDiagnosis distribution:")
        for diagnosis, count in sorted(diagnoses.items(), key=lambda x: x[1], reverse=True):
            print(f"  {diagnosis}: {count} samples")

def example_custom_processor():
    """Example of using individual processors."""
    print("\n=== Custom Processor Example ===")
    
    from ham10k_processor import HAM10KProcessor
    from pathlib import Path
    
    # Use a specific processor
    processor = HAM10KProcessor()
    dataset_path = Path("../../datasets/ham10k")
    
    if dataset_path.exists():
        try:
            processed_data = processor.process(dataset_path)
            print(f"✓ Processed {len(processed_data)} samples using HAM10KProcessor")
            
            # Display sample VQA questions
            if processed_data:
                sample = processed_data[0]
                print("\nSample VQA questions:")
                for i, qa in enumerate(sample.get('vqa_questions', [])[:3]):
                    print(f"  Q{i+1}: {qa['question']}")
                    print(f"  A{i+1}: {qa['answer']}")
                    print()
        except Exception as e:
            print(f"✗ Error processing with custom processor: {e}")
    else:
        print("✗ HAM10K dataset not found")

def main():
    """Run all examples."""
    print("Dermatology Metadata Preprocessor - Example Usage")
    print("=" * 50)
    
    # Create output directory
    Path("./example_output").mkdir(exist_ok=True)
    
    # Run examples
    example_single_dataset()
    example_all_datasets()
    example_combine_datasets()
    example_custom_processor()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the ./example_output directory for results.")

if __name__ == "__main__":
    main() 