#!/usr/bin/env python3
"""
Setup script for the Dermatology Metadata Preprocessor.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required dependencies."""
    print("Installing required dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating necessary directories...")
    
    directories = [
        "processed_metadata",
        "example_output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def verify_datasets():
    """Verify that datasets directory exists."""
    print("Verifying datasets directory...")
    
    datasets_dir = Path("../../datasets")
    if datasets_dir.exists():
        print("✓ Datasets directory found")
        
        # List available datasets
        available_datasets = []
        for item in datasets_dir.iterdir():
            if item.is_dir():
                available_datasets.append(item.name)
        
        if available_datasets:
            print(f"Available datasets: {', '.join(available_datasets)}")
        else:
            print("⚠ No datasets found in the datasets directory")
    else:
        print("⚠ Datasets directory not found at ../../datasets")
        print("Please ensure your datasets are located in the correct directory")

def run_example():
    """Ask user if they want to run an example."""
    print("\nSetup complete!")
    
    response = input("Would you like to run an example? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("Running example...")
        try:
            subprocess.run([sys.executable, "example_usage.py"])
        except Exception as e:
            print(f"Error running example: {e}")

def main():
    """Main setup function."""
    print("Dermatology Metadata Preprocessor Setup")
    print("=" * 40)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install dependencies manually.")
        return
    
    # Create directories
    create_directories()
    
    # Verify datasets
    verify_datasets()
    
    # Run example
    run_example()
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nUsage examples:")
    print("  python main.py --help")
    print("  python main.py --dataset ham10k")
    print("  python main.py --combine")
    print("  python example_usage.py")

if __name__ == "__main__":
    main() 