# LoanQuery AI

A modern, beautiful Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about loan approval patterns. Built with cutting-edge AI technologies and designed for ease of use. AI-Powered Loan Insight Assistant

- **Training Dataset.csv** file (place in project root)

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd LoanQuery-AI

# Place your Training Dataset.csv file in the project root directory

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Data

```bash
# Run one-time setup to prepare the data
python setup.py
```

This will:

- Load the Training Dataset.csv file from the project directory
- Download the embedding model (~90MB)
- Create FAISS vector index
- Process and store embeddings

### 3. Launch Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

