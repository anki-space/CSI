import requests
import json
from typing import List, Tuple, Generator
import time

class GroqLLM:
    def __init__(self, api_key: str = None):
        """Initialize Groq LLM interface."""
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = api_key or ""  # API key should be provided via environment variable or config
        self.model = "llama3-8b-8192"
        self.use_real_api = False  # Default to demo mode
    
    def set_api_key(self, api_key: str):
        """Set the API key dynamically."""
        self.api_key = api_key
        self.use_real_api = bool(api_key)
    
    def format_prompt(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """Format the prompt with context and query."""
        context_text = ""
        for i, (chunk, score) in enumerate(context_chunks, 1):
            context_text += f"{i}. {chunk}\n\n"
        
        prompt = f"""Context:
{context_text}

Question: {query}

Based on the context provided above, please answer the question. If the information needed to answer the question is not available in the context, please say so. Provide a clear, concise answer that is grounded in the data provided.

Answer:"""
        
        return prompt
    
    def generate_real_response(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """Generate response using actual Groq API."""
        try:
            prompt = self.format_prompt(query, context_chunks)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """Generate response using Groq API or mock responses."""
        
        # Use real API if configured and available
        if self.use_real_api and self.api_key:
            return self.generate_real_response(query, context_chunks)
        
        # Fallback to demo mode
        if not context_chunks:
            return "I don't have enough information to answer that question based on the available data."
        
        # Enhanced mock responses based on loan dataset patterns
        mock_responses = {
            "approval": "📊 Based on the loan data analysis, several key factors influence loan approval rates:\n\n• **Credit History**: The strongest predictor - applicants with established credit history show 80%+ approval rates\n• **Income Stability**: Higher and more stable income correlates with better approval chances\n• **Employment Type**: Salaried employees typically have higher approval rates than self-employed\n• **Debt-to-Income Ratio**: Lower ratios significantly improve approval likelihood\n• **Property Location**: Urban areas often show different approval patterns than rural areas",
            
            "denied": "❌ Common reasons for loan denials in the dataset include:\n\n• **Poor Credit History**: Missing or negative credit history is the top reason\n• **Insufficient Income**: Income too low relative to loan amount requested\n• **High Existing Debt**: Existing financial obligations affecting capacity\n• **Employment Instability**: Irregular income patterns, especially for self-employed\n• **Incomplete Documentation**: Missing required paperwork or verification\n• **Property Issues**: Problems with collateral or property valuation",
            
            "income": "💰 Income analysis reveals important patterns:\n\n• **Primary Income Impact**: Higher applicant income strongly correlates with approval\n• **Combined Income Effect**: Co-applicant income can significantly boost approval chances\n• **Income Thresholds**: Clear patterns emerge around certain income levels\n• **Stability Factor**: Consistent income history often more important than peak amounts\n• **Source Verification**: Documented, verifiable income streams crucial for approval",
            
            "self-employed": "🏢 Self-employed applicants face unique challenges:\n\n• **Documentation Requirements**: Need more extensive financial records\n• **Income Variability**: Fluctuating income creates uncertainty for lenders\n• **Lower Approval Rates**: Generally 15-20% lower than salaried employees\n• **Higher Scrutiny**: More detailed financial analysis required\n• **Mitigation Strategies**: Strong credit history and higher down payments help",
            
            "credit": "📈 Credit history analysis shows:\n\n• **Critical Factor**: Most important single predictor of loan approval\n• **Binary Impact**: Having vs. not having credit history creates stark differences\n• **Score Ranges**: Higher credit scores correlate with faster approval processes\n• **Historical Trends**: Consistent payment history valued over absolute scores\n• **Recovery Patterns**: Recent positive history can offset past issues",
            
            "urban": "🏙️ Property area analysis reveals:\n\n• **Urban Areas**: Higher approval rates, better property valuations\n• **Semi-Urban**: Moderate approval rates, mixed property dynamics\n• **Rural Areas**: Lower approval rates, property valuation challenges\n• **Market Factors**: Local economic conditions influence decisions\n• **Infrastructure**: Better connectivity and amenities support approvals"
        }
        
        # Enhanced keyword matching
        query_lower = query.lower()
        for keyword, response in mock_responses.items():
            if keyword in query_lower:
                return f"🤖 **Demo Mode Response:**\n\n{response}\n\n*Note: This is a sample response. Configure your Groq API key for real AI-powered analysis.*"
        
        # Default contextual response
        return f"""🤖 **Demo Mode Response:**

Based on the {len(context_chunks)} most relevant data points I found, here are some key insights:

• The loan dataset contains patterns across multiple dimensions including income, credit history, employment, and geography
• Approval decisions appear to be influenced by a combination of financial and demographic factors  
• There are clear differences in approval rates across different applicant segments
• Risk assessment seems to consider both quantitative metrics and qualitative factors

*For detailed, AI-powered analysis of your specific question, please configure your Groq API key in the settings above.*

**Sample insights from the data:**
- Credit history presence/absence significantly impacts outcomes
- Income levels and stability play crucial roles
- Employment type creates different risk profiles
- Property location affects approval patterns"""
    
    def generate_streaming_response(self, query: str, context_chunks: List[Tuple[str, float]]) -> Generator[str, None, None]:
        """Generate streaming response (simulated for demo)."""
        response = self.generate_response(query, context_chunks)
        
        # Simulate streaming by yielding words
        words = response.split()
        for word in words:
            yield word + " "
            time.sleep(0.05)  # Small delay to simulate streaming
