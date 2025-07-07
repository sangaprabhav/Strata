# Dermatology Metadata Preprocessor

This module provides scripts for preprocessing metadata from various dermatology datasets to create a standardized Visual Question Answering (VQA) dataset.

## Supported Datasets

The preprocessor supports the following datasets:

1. **BCN20K** - Large-scale dermatology dataset with comprehensive metadata
2. **DDI** - Diverse Dermatology Images dataset with skin tone diversity
3. **DDI2** - Diverse Dermatology Images dataset for Asian populations
4. **DERM12345** - Dermatology dataset with structured metadata
5. **HAM10K** - Human Against Machine with 10,000 training images
6. **HIBA** - Hospital Italiano de Buenos Aires skin lesion dataset
7. **ISIC2020** - International Skin Imaging Collaboration 2020 dataset
8. **MRA-MIDAS** - Melanoma Risk Assessment dataset
9. **MSKCC** - Memorial Sloan Kettering Cancer Center dataset
10. **PAD-UFES20** - Skin lesion dataset with patient demographics
11. **Patch16** - Patch-based classification dataset
12. **SCIN** - Skin Condition Image Network dataset

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Process All Datasets

```bash
python main.py
```

### Process a Specific Dataset

```bash
python main.py --dataset ham10k
```

### Combine All Processed Datasets

```bash
python main.py --combine
```

### Custom Directories

```bash
python main.py --datasets_dir /path/to/datasets --output_dir /path/to/output
```

## Output Format

Each processed dataset generates a JSON file with the following standardized structure:

```json
{
  "dataset_name": "HAM10K",
  "sample_id": "HAM_0000118",
  "image_path": "images/ISIC_0027419.jpg",
  "diagnosis": "nevus",
  "age": 45.0,
  "sex": "male",
  "anatomical_site": "trunk",
  "metadata": {
    "ham_id": "HAM_0000118",
    "isic_id": "ISIC_0027419",
    "diagnosis_type": "histo",
    "dataset_source": "rosendahl"
  },
  "vqa_questions": [
    {
      "question": "What is the diagnosis of this skin lesion?",
      "answer": "nevus"
    },
    {
      "question": "What is the age of the patient?",
      "answer": "45.0 years old"
    },
    {
      "question": "What is the sex of the patient?",
      "answer": "male"
    },
    {
      "question": "Where is this lesion located on the body?",
      "answer": "trunk"
    }
  ]
}
```

## Features

- **Standardized Output**: All datasets are converted to a common format
- **VQA Question Generation**: Automatically generates question-answer pairs
- **Robust Error Handling**: Gracefully handles missing or corrupted files
- **Flexible Configuration**: Configurable input/output directories
- **Batch Processing**: Process all datasets or specific ones
- **Data Validation**: Standardizes and validates metadata fields

## Standardization

The preprocessor standardizes the following fields:

- **Diagnosis**: Maps various diagnosis formats to standardized labels
- **Age**: Converts age values to numeric format
- **Sex**: Standardizes gender/sex information
- **Anatomical Site**: Maps body locations to standardized regions
- **Image Paths**: Creates consistent image path formats

## Adding New Datasets

To add a new dataset:

1. Create a new processor class inheriting from `BaseProcessor`
2. Implement the `process` method to handle the dataset's specific format
3. Add the processor to the main script's processor dictionary
4. Update the README with dataset information

Example:

```python
from .base_processor import BaseProcessor

class NewDatasetProcessor(BaseProcessor):
    def __init__(self):
        super().__init__('NewDataset')
    
    def process(self, dataset_path: Path) -> List[Dict[str, Any]]:
        # Implementation here
        pass
```

## Error Handling

The preprocessor includes robust error handling:

- Missing files are reported but don't stop processing
- Corrupted data is logged and skipped
- Summary reports show successful and failed processing

## Contributing

When contributing new processors:

1. Follow the existing code structure
2. Include appropriate error handling
3. Add dataset-specific VQA questions
4. Update documentation
5. Test with sample data

## License

This project is licensed under the same terms as the parent STRATA project. 