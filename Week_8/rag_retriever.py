import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from data_preprocessor import DataPreprocessor

class RAGRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG retriever."""
        self.model = SentenceTransformer(model_name)
        self.preprocessor = DataPreprocessor(model_name)
        self.index = None
        self.texts = None
        self.dataframe = None
        self.load_data()
    
    def load_data(self):
        """Load preprocessed data."""
        try:
            self.index, self.texts, self.dataframe = self.preprocessor.load_preprocessed_data()
            print(f"✅ Loaded {len(self.texts)} documents from FAISS index")
        except FileNotFoundError:
            print("❌ Preprocessed data not found. Running preprocessing...")
            if self.preprocessor.preprocess_data():
                self.index, self.texts, self.dataframe = self.preprocessor.load_preprocessed_data()
                print(f"✅ Loaded {len(self.texts)} documents from FAISS index")
            else:
                raise Exception("Failed to preprocess data")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant documents for a query."""
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                results.append((self.texts[idx], float(score)))
        
        return results
    
    def get_dataset_info(self) -> dict:
        """Get dataset overview information."""
        if self.dataframe is None:
            return {}
        
        info = {
            'total_rows': len(self.dataframe),
            'columns': list(self.dataframe.columns),
            'sample_rows': self.dataframe.head().to_dict('records'),
            'column_stats': {}
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            info['column_stats'][col] = {
                'mean': float(self.dataframe[col].mean()),
                'std': float(self.dataframe[col].std()),
                'min': float(self.dataframe[col].min()),
                'max': float(self.dataframe[col].max())
            }
        
        # Add value counts for categorical columns
        categorical_cols = self.dataframe.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            info['column_stats'][col] = dict(self.dataframe[col].value_counts().head())
        
        return info
