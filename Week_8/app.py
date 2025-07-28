import streamlit as st
import pandas as pd
from rag_retriever import RAGRetriever
from llm_interface import GroqLLM
import time
import os

# Page configuration
st.set_page_config(
    page_title="LoanQuery AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_setup_if_needed():
    """Run setup if FAISS index doesn't exist."""
    index_path = os.path.join("faiss_index", "loan_data.index")
    dataset_path = "Training Dataset.csv"
    
    if not os.path.exists(index_path) or not os.path.exists(dataset_path):
        st.error("⚠️ Setup required! Please run the setup first.")
        
        with st.expander("🛠️ Setup Instructions", expanded=True):
            st.markdown("""
            **Missing required files:**
            - `Training Dataset.csv` (place in project root)
            - FAISS index files (generated during setup)
            
            **To set up locally:**
            ```bash
            python setup.py
            ```
            
            **For Streamlit Cloud deployment:**
            The setup should run automatically, but if it fails:
            1. Ensure `Training Dataset.csv` is in your repository
            2. Check that all dependencies are properly installed
            3. Try restarting the app
            """)
        
        if st.button("🚀 Try Auto Setup"):
            try:
                with st.spinner("Running setup... This may take a few minutes."):
                    # Try to run setup automatically
                    from setup import main as setup_main
                    setup_main()
                    st.success("✅ Setup completed! Please refresh the page.")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Setup failed: {str(e)}")
                st.info("Please run `python setup.py` locally first.")
        
        return False
    return True

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Background and global font */
    .stApp {
        background: radial-gradient(ellipse at center, #0d0d0d 0%, #050505 100%);
        color: #dbeafe;
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.125rem;
        margin-bottom: 2rem;
    }

    /* Frosted glass chat container */
    .chat-container {
        background: rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }

    /* Chat messages */
    .chat-message {
        padding: 1.25rem;
        border-radius: 1rem;
        margin: 1rem 0;
        animation: fadeInUp 0.3s ease-out;
    }

    .user-message {
        background: linear-gradient(to right, #1e3a8a, #1e40af);
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 4px;
    }

    .assistant-message {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        color: white;
        margin-right: 2rem;
        border-bottom-left-radius: 4px;
    }

    /* Sidebar box */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        width: 100%;
        box-shadow: 0 3px 8px rgba(59, 130, 246, 0.3);
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #60a5fa, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08);
        color: #e0f2fe;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.875rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }

    /* Source box */
    .source-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #cbd5e1;
    }

    /* API key box */
    .api-key-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar transparent fix */
    .css-1d391kg {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_rag_system():
    """Load and cache the RAG retriever system."""
    return RAGRetriever()

@st.cache_resource
def load_llm():
    """Load and cache the LLM interface."""
    return GroqLLM()

def display_dataset_overview(retriever):
    """Display dataset overview in the sidebar."""
    with st.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("## 📊 Dataset Overview")
        
        try:
            dataset_info = retriever.get_dataset_info()
            
            if dataset_info:
                st.metric("Total Records", dataset_info['total_rows'])
                st.metric("Features", len(dataset_info['columns']))
                
                # Display column information
                st.markdown("### 📋 Columns")
                col_text = ""
                for col in dataset_info['columns']:
                    col_text += f"• {col}  \n"
                st.markdown(col_text)
                
                # Display key statistics
                st.markdown("### 📈 Key Stats")
                if 'Loan_Status' in dataset_info['column_stats']:
                    loan_stats = dataset_info['column_stats']['Loan_Status']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Approved", loan_stats.get('Y', 0))
                    with col2:
                        st.metric("Denied", loan_stats.get('N', 0))
        
        except Exception as e:
            st.error(f"Error loading dataset info: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_chat_message(role, content, avatar="🤖"):
    """Display a chat message with styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>{avatar} LoanQuery AI:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_sources(sources):
    """Display source information."""
    if sources:
        st.markdown("### 📚 Source Information")
        st.markdown("*The following data points were used to generate the response:*")
        
        for i, (source, score) in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i} (Relevance: {score:.3f})</strong><br>
                {source}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Run setup if needed
    if not run_setup_if_needed():
        return
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LoanQuery AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Get AI-powered insights about loan approval patterns using advanced RAG technology</p>', unsafe_allow_html=True)
    
    # API Key Configuration
    with st.expander("⚙️ API Configuration", expanded=False):
        st.markdown('<div class="api-key-section">', unsafe_allow_html=True)
        st.markdown("### 🔑 Groq API Key")
        st.markdown("Enter your Groq API key to enable real AI responses (optional - demo mode available)")
        
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your Groq API key here...",
            help="Get your free API key from https://console.groq.com"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            use_demo_mode = st.checkbox("Use Demo Mode", value=True, help="Use mock responses for testing")
        with col2:
            if st.button("Test Connection"):
                if api_key:
                    st.success("✅ API key configured!")
                else:
                    st.info("💡 Using demo mode with mock responses")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load systems
    with st.spinner("🔄 Loading RAG system..."):
        retriever = load_rag_system()
        llm = load_llm()
        
        # Update LLM with API key if provided
        if api_key and not use_demo_mode:
            llm.api_key = api_key
            llm.use_real_api = True
        else:
            llm.use_real_api = False
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("## 💬 Chat Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = """👋 Welcome! I'm LoanQuery AI, your intelligent assistant for loan analysis. 

I can analyze loan approval patterns and answer questions like:
• Why are some applications denied?
• What factors influence approval rates?
• How does income affect loan decisions?
• Are there differences by property area?

**Try asking me anything about the loan data!**"""
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Chat input
        if prompt := st.chat_input("💭 Ask me anything about loan approvals..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # Process query
            with st.spinner("🔍 Analyzing data..."):
                # Retrieve relevant context
                sources = retriever.retrieve(prompt, top_k=5)
                
                # Generate response
                response = llm.generate_response(prompt, sources)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                # Display sources
                if sources:
                    with st.expander("📚 View Source Data", expanded=False):
                        display_sources(sources)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar content
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("## 🔧 Quick Actions")
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📊 Show Sample Data"):
            dataset_info = retriever.get_dataset_info()
            if dataset_info and dataset_info['sample_rows']:
                st.dataframe(pd.DataFrame(dataset_info['sample_rows']).head())
        
        # Sample questions
        st.markdown("## 💡 Try These Questions")
        
        sample_questions = [
            "What factors affect loan approval?",
            "Why are loans denied?",
            "How does credit history impact approval?",
            "What's the approval rate by property area?",
            "Do self-employed applicants get approved less?",
            "How does income affect loan decisions?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}"):
                # Add the question to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Process the question
                sources = retriever.retrieve(question, top_k=5)
                response = llm.generate_response(question, sources)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset overview
        display_dataset_overview(retriever)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 2rem;">
        <p>🔒 <strong>Privacy First:</strong> All processing happens locally on your machine</p>
        <p>📊 <strong>Powered by:</strong> RAG Architecture + FAISS + Sentence Transformers</p>
        <p>🚀 <strong>Built with:</strong> Streamlit + Python</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
