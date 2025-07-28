#!/usr/bin/env python3
"""
Setup script for LoanQuery AI
This script prepares the data and embeddings for the chatbot.
"""

import os
import sys
from data_preprocessor import DataPreprocessor

def main():
    print("🚀 Setting up LoanQuery AI...")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    print("📊 Starting data preprocessing...")
    success = preprocessor.preprocess_data()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\n🎉 Your LoanQuery AI is ready!")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
        print("\nThe app will be available at: http://localhost:8501")
    else:
        print("\n❌ Setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
