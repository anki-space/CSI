import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class DataPreprocessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the data preprocessor with embedding model."""
        self.model = SentenceTransformer(model_name)
        self.index_dir = "faiss_index"
        self.index_path = os.path.join(self.index_dir, "loan_data.index")
        self.metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        
    def download_dataset(self) -> pd.DataFrame:
        """Load the loan dataset from the Training Dataset.csv file."""
        try:
            print("Loading Training Dataset.csv...")
            
            # Check if the file exists
            csv_path = "Training Dataset.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Training Dataset.csv not found in the current directory. Please ensure the file is placed in the project root directory.")
            
            # Load the CSV file with proper handling for missing values
            df = pd.read_csv(csv_path, na_values=['', ' ', 'NA', 'na', 'NULL', 'null'])
            print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Display basic info about the dataset
            print("Dataset columns:", list(df.columns))
            print("Dataset shape:", df.shape)
            print("Missing values per column:")
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  {col}: {missing_count} missing values")
            
            return df
            
        except Exception as e:
            print(f"Error loading Training Dataset.csv: {str(e)}")
            print("Please ensure 'Training Dataset.csv' is in the project root directory.")
            raise
    
    def row_to_text(self, row: pd.Series) -> str:
        """Convert a dataframe row to natural language text."""
        # Get loan ID
        loan_id = str(row.get('Loan_ID', f'Unknown_{row.name}')).strip()
        
        # Handle gender
        gender = str(row.get('Gender', '')).strip()
        if pd.isna(row.get('Gender')) or gender == '':
            gender = 'unknown gender'
        else:
            gender = gender.lower()
        
        # Handle marital status
        married_status = str(row.get('Married', '')).strip()
        married = 'married' if married_status.lower() == 'yes' else 'not married'
        
        # Handle dependents
        dependents = str(row.get('Dependents', '')).strip()
        if pd.isna(row.get('Dependents')) or dependents == '':
            dependents = 'unknown number of'
        elif dependents == '3+':
            dependents = '3 or more'
        else:
            dependents = dependents
            
        # Handle education
        education = str(row.get('Education', '')).strip()
        if pd.isna(row.get('Education')) or education == '':
            education = 'unknown education level'
        else:
            education = education.lower()
            
        # Handle self employment
        self_emp_status = str(row.get('Self_Employed', '')).strip()
        self_employed = 'self-employed' if self_emp_status.lower() == 'yes' else 'not self-employed'
        
        # Handle credit history
        credit_val = row.get('Credit_History')
        if pd.isna(credit_val):
            credit_history = 'unknown credit history'
        else:
            try:
                credit_history = 'has credit history' if float(credit_val) == 1.0 else 'no credit history'
            except (ValueError, TypeError):
                credit_history = 'unknown credit history'
        
        # Handle property area
        property_area = str(row.get('Property_Area', '')).strip()
        if pd.isna(row.get('Property_Area')) or property_area == '':
            property_area = 'unknown'
        else:
            property_area = property_area.lower()
            
        # Handle loan status
        loan_status_val = str(row.get('Loan_Status', '')).strip()
        loan_status = 'approved' if loan_status_val.upper() == 'Y' else 'denied'
        
        # Handle numeric values with proper missing value handling
        loan_amount = row.get('LoanAmount')
        if pd.isna(loan_amount):
            loan_amount_text = 'an unspecified amount'
        else:
            try:
                loan_amount_text = f"{float(loan_amount)} thousand"
            except (ValueError, TypeError):
                loan_amount_text = 'an unspecified amount'
            
        applicant_income = row.get('ApplicantIncome')
        if pd.isna(applicant_income):
            applicant_income_text = 'unspecified income'
        else:
            try:
                applicant_income_text = f"income of {float(applicant_income)}"
            except (ValueError, TypeError):
                applicant_income_text = 'unspecified income'
            
        coapplicant_income = row.get('CoapplicantIncome')
        if pd.isna(coapplicant_income) or coapplicant_income == 0:
            coapplicant_income_text = 'no co-applicant income'
        else:
            try:
                coapplicant_income_text = f"co-applicant income of {float(coapplicant_income)}"
            except (ValueError, TypeError):
                coapplicant_income_text = 'unspecified co-applicant income'
        
        # Handle loan term
        loan_term = row.get('Loan_Amount_Term')
        if pd.isna(loan_term):
            loan_term_text = 'unspecified term'
        else:
            try:
                loan_term_text = f"{int(float(loan_term))} months term"
            except (ValueError, TypeError):
                loan_term_text = 'unspecified term'
        
        # Construct the descriptive text
        text = (f"Applicant {loan_id} is a {married} {gender} {education} who is {self_employed} "
                f"with {dependents} dependents and {credit_history}. "
                f"They applied for a loan of {loan_amount_text} with {applicant_income_text} "
                f"and {coapplicant_income_text} for {loan_term_text} in a {property_area} area. "
                f"The loan was {loan_status}.")
        
        return text
    
    def preprocess_data(self) -> bool:
        """Complete preprocessing pipeline: load data, convert to text, embed, and store."""
        try:
            # Create index directory if it doesn't exist
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Load dataset first to check if we need to reprocess
            df = self.download_dataset()
            print(f"Loaded dataset with {len(df)} rows")
            
            # Check if already processed and if the data matches
            regenerate = False
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    stored_df = metadata.get('dataframe')
                    
                    # Check if the dataset has changed
                    if stored_df is None or len(stored_df) != len(df) or not stored_df.equals(df):
                        print("Dataset has changed. Regenerating embeddings...")
                        regenerate = True
                    else:
                        print("Preprocessed data already exists and matches current dataset. Skipping preprocessing.")
                        return True
                except Exception as e:
                    print(f"Error reading existing metadata: {e}. Regenerating...")
                    regenerate = True
            else:
                regenerate = True
            
            if regenerate:
                print("Starting data preprocessing...")
                
                # Convert rows to text
                texts = []
                for idx, row in df.iterrows():
                    text = self.row_to_text(row)
                    texts.append(text)
                
                print(f"Converted {len(texts)} rows to text")
                
                # Show a sample of the converted text for verification
                if texts:
                    print("\nSample converted text:")
                    print(f"  {texts[0]}")
                    if len(texts) > 1:
                        print(f"  {texts[1]}")
                
                # Generate embeddings
                print("\nGenerating embeddings...")
                embeddings = self.model.encode(texts, show_progress_bar=True)
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                index.add(embeddings.astype('float32'))
                
                # Save index
                faiss.write_index(index, self.index_path)
                
                # Save metadata (original texts and dataframe)
                metadata = {
                    'texts': texts,
                    'dataframe': df
                }
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                print(f"\nPreprocessing complete! Saved {len(texts)} embeddings to {self.index_path}")
            
            return True
            
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            return False
    
    def load_preprocessed_data(self) -> Tuple[faiss.Index, List[str], pd.DataFrame]:
        """Load the preprocessed FAISS index and metadata."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Preprocessed data not found. Run preprocess_data() first.")
        
        # Load FAISS index
        index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata['texts'], metadata['dataframe']

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = DataPreprocessor()
    success = preprocessor.preprocess_data()
    if success:
        print("✅ Preprocessing completed successfully!")
    else:
        print("❌ Preprocessing failed!")
