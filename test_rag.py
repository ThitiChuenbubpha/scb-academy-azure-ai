import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# Load environment variables
load_dotenv()


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents. 
Use only the information from the context to answer questions. If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the source document and page number when referencing information. ตอบภาษาไทยเท่านั้น ห้ามตอบอังกฤษ ถึงแม้คำถามจะเป็นภาษาอังกฤษ"""


class RAGConfig:
    """Configuration class for RAG system"""
    def __init__(self):
        # Azure OpenAI Configuration
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.embedding_deployment = "text-embedding-3-large"
        self.chat_deployment = "gpt-4.1-mini"
        
        # Azure AI Search Configuration
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        # LLM Configuration
        self.temperature = 0.7 #0.0 to 1.0
        self.max_tokens = 4096 #1 to 4096
        self.top_p = 0.9 #0.0 to 1.0
        self.frequency_penalty = 0.0 #-2.0 to 2.0
        self.presence_penalty = 0.0 #-2.0 to 2.0
        
        # Search Configuration
        self.top_k_documents = 5
        self.use_hybrid_search = False
        self.hybrid_search_weight = 0.5  # Weight for semantic vs keyword search in hybrid mode

class RAGSystem:
    """RAG system for testing with Azure OpenAI and Azure AI Search"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=config.azure_openai_key,
            api_version="2024-02-15-preview",
            azure_endpoint=config.azure_openai_endpoint
        )
        
        # Initialize embedding client
        self.embedding_client = AzureOpenAI(
            api_key=config.azure_openai_key,
            api_version="2024-02-15-preview",
            azure_endpoint=config.embedding_endpoint
        )
        
        # Initialize search client
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index_name,
            credential=AzureKeyCredential(config.search_key)
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text"""
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self.config.embedding_deployment,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            if self.config.use_hybrid_search:
                return self._hybrid_search(query)
            else:
                return self._vector_search(query)
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform vector search"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self.config.top_k_documents,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=self.config.top_k_documents,
            select=["id", "content", "source_filename", "page_number"]
        )
        
        documents = []
        for result in results:
            documents.append({
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "source_filename": result.get("source_filename", ""),
                "page_number": result.get("page_number", 0),
                "score": result.get("@search.score", 0.0)
            })
        
        return documents
    
    def _hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform hybrid search (vector + keyword)"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self.config.top_k_documents,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=self.config.top_k_documents,
            select=["id", "content", "source_filename", "page_number"]
        )
        
        documents = []
        for result in results:
            documents.append({
                "id": result.get("id", ""),
                "content": result.get("content", ""),
                "source_filename": result.get("source_filename", ""),
                "page_number": result.get("page_number", 0),
                "score": result.get("@search.score", 0.0)
            })
        
        return documents
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with retrieved context"""
        if not context_documents:
            return "No relevant documents found to answer your question."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source_info = f"[Source: {doc['source_filename']}, Page: {doc['page_number']}]"
            context_parts.append(f"Document {i} {source_info}:\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt
        system_prompt = SYSTEM_PROMPT
        
        user_prompt = f"""Context Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context documents above."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query method that performs RAG"""
        print(f"\n🔍 Searching for: '{question}'")
        print(f"Search mode: {'Hybrid' if self.config.use_hybrid_search else 'Vector'}")
        print(f"Top-K documents: {self.config.top_k_documents}")
        
        # Search for relevant documents
        documents = self.search_documents(question)
        
        print(f"📄 Found {len(documents)} relevant documents")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc['source_filename']} (Page {doc['page_number']}) - Score: {doc['score']:.4f}")
        
        # Generate response
        print("\n🤖 Generating response...")
        response = self.generate_response(question, documents)
        
        return {
            "question": question,
            "response": response,
            "retrieved_documents": documents,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "top_k_documents": self.config.top_k_documents,
                "use_hybrid_search": self.config.use_hybrid_search
            }
        }

def interactive_config():
    """Interactive configuration setup"""
    config = RAGConfig()
    
    print("🔧 RAG System Configuration")
    print("=" * 40)
    
    # LLM Configuration
    print("\n📝 LLM Configuration:")
    temp_input = input(f"Temperature (current: {config.temperature}): ").strip()
    if temp_input:
        config.temperature = float(temp_input)
    
    tokens_input = input(f"Max tokens (current: {config.max_tokens}): ").strip()
    if tokens_input:
        config.max_tokens = int(tokens_input)
    
    top_p_input = input(f"Top-P (current: {config.top_p}): ").strip()
    if top_p_input:
        config.top_p = float(top_p_input)
    
    freq_penalty_input = input(f"Frequency penalty (current: {config.frequency_penalty}): ").strip()
    if freq_penalty_input:
        config.frequency_penalty = float(freq_penalty_input)
    
    pres_penalty_input = input(f"Presence penalty (current: {config.presence_penalty}): ").strip()
    if pres_penalty_input:
        config.presence_penalty = float(pres_penalty_input)
    
    # Search Configuration
    print("\n🔍 Search Configuration:")
    top_k_input = input(f"Number of documents to retrieve (current: {config.top_k_documents}): ").strip()
    if top_k_input:
        config.top_k_documents = int(top_k_input)
    
    hybrid_input = input(f"Use hybrid search? y/N (current: {'Yes' if config.use_hybrid_search else 'No'}): ").strip().lower()
    if hybrid_input in ['y', 'yes']:
        config.use_hybrid_search = True
    elif hybrid_input in ['n', 'no']:
        config.use_hybrid_search = False
    
    return config

def main():
    """Main function for interactive RAG testing"""
    print("🚀 Welcome to RAG Testing System")
    print("=" * 50)
    
    # Configuration
    config_choice = input("Do you want to configure settings? (y/N): ").strip().lower()
    if config_choice in ['y', 'yes']:
        config = interactive_config()
    else:
        config = RAGConfig()
        print("Using default configuration...")
    
    # Initialize RAG system
    try:
        rag_system = RAGSystem(config)
        print("\n✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Interactive query loop
    print("\n💬 Interactive Query Mode")
    print("Type 'quit' to exit, 'config' to reconfigure")
    print("-" * 50)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif question.lower() == 'config':
            config = interactive_config()
            rag_system = RAGSystem(config)
            print("✅ Configuration updated!")
            continue
        elif not question:
            continue
        
        # Process query
        try:
            result = rag_system.query(question)
            print(f"\n💡 Response:")
            print("-" * 30)
            print(result["response"])
            print("-" * 30)
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main() 