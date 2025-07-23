import streamlit as st
import os
import tempfile
import shutil
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import re
import unicodedata
import datetime
import gc
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_DPI = 300
MODEL_NAME = "gpt-4.1-mini"
API_VERSION = "2024-02-15-preview"
MAX_TOKENS = 4096
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
BATCH_SIZE = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents. 
Use only the information from the context to answer questions. If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the source document and page number when referencing information. ตอบภาษาไทยเท่านั้น ห้ามตอบอังกฤษ ถึงแม้คำถามจะเป็นภาษาอังกฤษ"""

class DocumentProcessor:
    def __init__(self, credentials: Optional[Dict[str, str]] = None, config_params: Optional[Dict[str, Any]] = None):
        self.credentials = credentials or {}
        self.config = config_params or {}
        self.azure_openai_client = self._setup_azure_client()
        self.search_client = self._setup_search_client()
        
    def _get_credential(self, key: str) -> str:
        """Get credential from user input or environment variables."""
        return self.credentials.get(key) or os.getenv(key, "")
        
    def _setup_azure_client(self) -> AzureOpenAI:
        """Set up and return Azure OpenAI client."""
        endpoint = self._get_credential("AZURE_OPENAI_ENDPOINT")
        api_key = self._get_credential("AZURE_OPENAI_KEY")
        
        if not endpoint or not api_key:
            raise ValueError(
                "Missing Azure OpenAI credentials: endpoint and API key are required"
            )
        
        return AzureOpenAI(
            api_key=api_key,
            api_version=API_VERSION,
            azure_endpoint=endpoint
        )
    
    def _setup_search_client(self) -> SearchClient:
        """Set up and return Azure Search client."""
        search_endpoint = self._get_credential("AZURE_SEARCH_ENDPOINT")
        search_key = self._get_credential("AZURE_SEARCH_KEY")
        search_index_name = self._get_credential("AZURE_SEARCH_INDEX_NAME")
        
        if not all([search_endpoint, search_key, search_index_name]):
            raise ValueError(
                "Missing Azure Search credentials: endpoint, key, and index name are required"
            )
        
        return SearchClient(
            endpoint=search_endpoint,
            index_name=search_index_name,
            credential=AzureKeyCredential(search_key)
        )
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[bytes]:
        """Convert PDF pages to PNG images."""
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            # Calculate zoom for desired DPI
            image_dpi = self.config.get("image_dpi", IMAGE_DPI)
            zoom = image_dpi / 72  # 72 is default DPI
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pixmap = page.get_pixmap(matrix=matrix)
                img_data = pixmap.tobytes("png")
                images.append(img_data)
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using Azure OpenAI Vision API."""
        # Convert image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Instruction for the AI model
        prompt = (
            "You are an expert document extractor. Extract all useful text content "
            "from this image for creating a vector database for RAG applications. "
            "Include headers, bullet points, and all text elements. For structured "
            "information like tables, diagrams, or graphs, describe them in text "
            "with as much detail as possible, If the graph or chart have labels try to extract information on chart of each label."
            "Skip non-textual elements like QR codes. "
            "Output only useful content in a clear, human-readable format without "
            "meta-commentary or formatting markers."
        )
        
        try:
            response = self.azure_openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.get("max_tokens", MAX_TOKENS)
            )
            return response.choices[0].message.content or ""
            
        except Exception as e:
            error_msg = f"Error extracting text from image: {e}"
            logger.error(error_msg)
            return error_msg
    
    def process_pdf_to_text(self, pdf_path: str) -> str:
        """Process a single PDF file and extract text."""
        try:
            # Convert PDF to images
            images = self.convert_pdf_to_images(pdf_path)
            if not images:
                return "No images extracted from PDF"
            
            # Extract text from each image
            extracted_pages = []
            for i, image_data in enumerate(images, 1):
                text = self.extract_text_from_image(image_data)
                extracted_pages.append(f"\n--- Page {i} ---\n{text}")
            
            return '\n'.join(extracted_pages)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return f"Error processing PDF: {e}"
    
    def clean_text(self, text: str) -> str:
        """Clean text for embedding."""
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if char.isprintable())
        text = text.strip()
        return text
    
    def get_embedding(self, text: str):
        """Get embedding for text."""
        cleaned_text = self.clean_text(text)
        if len(cleaned_text.strip()) < 10:
            return None
        
        embedding_endpoint = self._get_credential("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        embedding_client = AzureOpenAI(
            api_key=self._get_credential("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=embedding_endpoint
        )
        
        response = embedding_client.embeddings.create(
            input=cleaned_text,
            model=EMBEDDING_DEPLOYMENT,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def split_text_into_chunks(self, text: str):
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        chunk_size = self.config.get("chunk_size", CHUNK_SIZE)
        chunk_overlap = self.config.get("chunk_overlap", CHUNK_OVERLAP)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Extract page numbers from chunks
        page_numbers = []
        for chunk in chunks:
            page_match = re.search(r'--- Page (\d+) ---', chunk)
            if page_match:
                page_numbers.append(int(page_match.group(1)))
            else:
                # If no page marker found, assume page 1
                page_numbers.append(1)
        
        return chunks, page_numbers
    
    def upload_to_search_index(self, text: str, filename: str) -> bool:
        """Upload processed text to Azure AI Search."""
        try:
            chunks, page_numbers = self.split_text_into_chunks(text)
            
            # Sanitize document ID
            sanitized_doc_id = re.sub(r'[^a-zA-Z0-9-]', '_', os.path.splitext(filename)[0])
            if not sanitized_doc_id or not sanitized_doc_id[0].isalnum():
                sanitized_doc_id = 'doc_' + sanitized_doc_id
            
            docs_to_upload = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.get_embedding(chunk)
                    if embedding is None:
                        continue
                    
                    doc = {
                        "id": f"{sanitized_doc_id}_{i}",
                        "content": self.clean_text(chunk),
                        "content_vector": embedding,
                        "source_filename": filename,
                        "last_modified": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "page_number": page_numbers[i],
                    }
                    docs_to_upload.append(doc)
                    
                    # Upload in batches
                    batch_size = self.config.get("batch_size", BATCH_SIZE)
                    if len(docs_to_upload) >= batch_size:
                        self.search_client.upload_documents(docs_to_upload)
                        docs_to_upload = []
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Upload remaining documents
            if docs_to_upload:
                self.search_client.upload_documents(docs_to_upload)
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to search index: {e}")
            return False

class RAGChatbot:
    def __init__(self, credentials: Optional[Dict[str, str]] = None, config_params: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None):
        self.credentials = credentials or {}
        self.config = config_params or {}
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.azure_openai_client = self._setup_azure_client()
        self.search_client = self._setup_search_client()
        
    def _get_credential(self, key: str) -> str:
        """Get credential from user input or environment variables."""
        return self.credentials.get(key) or os.getenv(key, "")
        
    def _setup_azure_client(self) -> AzureOpenAI:
        """Set up and return Azure OpenAI client."""
        endpoint = self._get_credential("AZURE_OPENAI_ENDPOINT")
        api_key = self._get_credential("AZURE_OPENAI_KEY")
        
        return AzureOpenAI(
            api_key=api_key,
            api_version=API_VERSION,
            azure_endpoint=endpoint
        )
    
    def _setup_search_client(self) -> SearchClient:
        """Set up and return Azure Search client."""
        search_endpoint = self._get_credential("AZURE_SEARCH_ENDPOINT")
        search_key = self._get_credential("AZURE_SEARCH_KEY")
        search_index_name = self._get_credential("AZURE_SEARCH_INDEX_NAME")
        
        return SearchClient(
            endpoint=search_endpoint,
            index_name=search_index_name,
            credential=AzureKeyCredential(search_key)
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text"""
        try:
            embedding_endpoint = self._get_credential("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            embedding_client = AzureOpenAI(
                api_key=self._get_credential("AZURE_OPENAI_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=embedding_endpoint
            )
            
            response = embedding_client.embeddings.create(
                input=text,
                model=EMBEDDING_DEPLOYMENT,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def search_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            if top_k is None:
                top_k = self.config.get("top_k_documents", 5)
                
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top_k,
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
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with retrieved context"""
        if not context_documents:
            return "ไม่พบเอกสารที่เกี่ยวข้องเพื่อตอบคำถามของคุณ"
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            source_info = f"[Source: {doc['source_filename']}, Page: {doc['page_number']}]"
            context_parts.append(f"Document {i} {source_info}:\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        user_prompt = f"""Context Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context documents above."""
        
        try:
            response = self.azure_openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 4096),
                top_p=self.config.get("top_p", 0.9),
                frequency_penalty=self.config.get("frequency_penalty", 0.0),
                presence_penalty=self.config.get("presence_penalty", 0.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}"
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Main chat method that performs RAG"""
        # Search for relevant documents
        documents = self.search_documents(question)
        
        # Generate response
        response = self.generate_response(question, documents)
        
        return {
            "question": question,
            "response": response,
            "retrieved_documents": documents
        }

def main():
    st.set_page_config(
        page_title="PDF RAG Chat System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF RAG Chat System")
    st.markdown("Upload PDF documents, process them, and chat with your documents using RAG!")
    
    # Sidebar for Azure credentials
    with st.sidebar:
        st.header("🔑 Azure Credentials")
        st.markdown("Enter your Azure OpenAI and Azure AI Search credentials:")
        
        # Azure OpenAI credentials
        st.subheader("Azure OpenAI")
        azure_openai_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            placeholder="https://your-openai-resource.openai.azure.com/",
            help="Your Azure OpenAI service endpoint"
        )
        
        azure_openai_key = st.text_input(
            "Azure OpenAI API Key",
            value=os.getenv("AZURE_OPENAI_KEY", ""),
            type="password",
            help="Your Azure OpenAI API key"
        )
        
        azure_embedding_endpoint = st.text_input(
            "Azure OpenAI Embedding Endpoint",
            value=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
            placeholder="https://your-openai-resource.openai.azure.com/",
            help="Your Azure OpenAI embedding endpoint (can be same as above)"
        )
        
        # Azure AI Search credentials
        st.subheader("Azure AI Search")
        azure_search_endpoint = st.text_input(
            "Azure Search Endpoint",
            value=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            placeholder="https://your-search-service.search.windows.net",
            help="Your Azure AI Search service endpoint"
        )
        
        azure_search_key = st.text_input(
            "Azure Search Admin Key",
            value=os.getenv("AZURE_SEARCH_KEY", ""),
            type="password",
            help="Your Azure AI Search admin key"
        )
        
        azure_search_index = st.text_input(
            "Azure Search Index Name",
            value=os.getenv("AZURE_SEARCH_INDEX_NAME", ""),
            placeholder="your-index-name",
            help="The name of your Azure AI Search index"
        )
        
        # Validate credentials
        credentials_valid = all([
            azure_openai_endpoint,
            azure_openai_key,
            azure_embedding_endpoint,
            azure_search_endpoint,
            azure_search_key,
            azure_search_index
        ])
        
        if credentials_valid:
            st.success("✅ All credentials provided!")
            
            # Store credentials in session state
            st.session_state.azure_credentials = {
                "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
                "AZURE_OPENAI_KEY": azure_openai_key,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": azure_embedding_endpoint,
                "AZURE_SEARCH_ENDPOINT": azure_search_endpoint,
                "AZURE_SEARCH_KEY": azure_search_key,
                "AZURE_SEARCH_INDEX_NAME": azure_search_index
            }
        else:
            st.error("❌ Please provide all required credentials")
            st.info("💡 You can also set these as environment variables in a .env file")
            
        st.markdown("---")
        
        # Configuration Section
        st.header("⚙️ Configuration")
        
        # System Prompt Configuration
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PROMPT,
            height=150,
            help="Define how the AI assistant should behave and respond to questions"
        )
        
        # Store system prompt in session state
        st.session_state.system_prompt = system_prompt
        
        # LLM Parameters
        st.subheader("LLM Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in responses (0 = deterministic, 1 = creative)"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4096,
                value=4096,
                step=100,
                help="Maximum length of generated responses"
            )
            
            top_p = st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Controls diversity via nucleus sampling"
            )
        
        with col2:
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Reduces repetition of tokens based on frequency"
            )
            
            presence_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Reduces repetition of tokens based on presence"
            )
            
            top_k_documents = st.number_input(
                "Top-K Documents",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Number of relevant documents to retrieve for context"
            )
        
        # Document Processing Parameters
        st.subheader("Document Processing")
        
        col3, col4 = st.columns(2)
        
        with col3:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=100,
                step=50,
                help="Overlap between consecutive chunks"
            )
        
        with col4:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Number of documents to upload at once"
            )
            
            image_dpi = st.number_input(
                "Image DPI",
                min_value=72,
                max_value=600,
                value=300,
                step=50,
                help="Resolution for PDF to image conversion"
            )
        
        # Store all parameters in session state
        st.session_state.config_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_k_documents": top_k_documents,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "batch_size": batch_size,
            "image_dpi": image_dpi
        }
        
        # Reset to defaults button
        if st.button("🔄 Reset to Defaults"):
            # Reset all session state config variables
            if 'config_params' in st.session_state:
                del st.session_state.config_params
            if 'system_prompt' in st.session_state:
                del st.session_state.system_prompt
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Need help?**")
        st.markdown("- [Azure OpenAI Setup Guide](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)")
        st.markdown("- [Azure AI Search Setup Guide](https://docs.microsoft.com/en-us/azure/search/)")
    
    # Check if credentials are available
    if not credentials_valid:
        st.warning("⚠️ Please provide your Azure credentials in the sidebar to get started.")
        st.info("""
        **Required Services:**
        - **Azure OpenAI**: For text extraction and chat responses
        - **Azure AI Search**: For storing and searching document chunks
        
        **Required Models:**
        - GPT-4 Vision (for PDF text extraction)
        - text-embedding-3-large (for embeddings)
        """)
        return
    
    # Initialize processors with user credentials and configuration
    try:
        # Check if we need to reinitialize due to credential or config changes
        current_creds = st.session_state.azure_credentials
        current_config = st.session_state.config_params
        current_prompt = st.session_state.system_prompt
        
        # Check if anything has changed
        need_reinit = (
            'processor' not in st.session_state or 
            'chatbot' not in st.session_state or
            'last_credentials' not in st.session_state or 
            'last_config' not in st.session_state or
            'last_prompt' not in st.session_state or
            st.session_state.last_credentials != current_creds or
            st.session_state.last_config != current_config or
            st.session_state.last_prompt != current_prompt
        )
        
        if need_reinit:
            st.session_state.processor = DocumentProcessor(current_creds, current_config)
            st.session_state.chatbot = RAGChatbot(current_creds, current_config, current_prompt)
            st.session_state.last_credentials = current_creds.copy()
            st.session_state.last_config = current_config.copy()
            st.session_state.last_prompt = current_prompt
            
    except Exception as e:
        st.error(f"❌ Error initializing services: {e}")
        st.info("Please check your credentials and try again.")
        return
    
    # Show current configuration status
    with st.expander("📊 Current Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temperature", f"{st.session_state.config_params['temperature']:.1f}")
            st.metric("Max Tokens", st.session_state.config_params['max_tokens'])
            st.metric("Top-K Documents", st.session_state.config_params['top_k_documents'])
        
        with col2:
            st.metric("Top-P", f"{st.session_state.config_params['top_p']:.1f}")
            st.metric("Freq. Penalty", f"{st.session_state.config_params['frequency_penalty']:.1f}")
            st.metric("Chunk Size", st.session_state.config_params['chunk_size'])
        
        with col3:
            st.metric("Presence Penalty", f"{st.session_state.config_params['presence_penalty']:.1f}")
            st.metric("Batch Size", st.session_state.config_params['batch_size'])
            st.metric("Image DPI", st.session_state.config_params['image_dpi'])
        
        st.text_area("Current System Prompt", st.session_state.system_prompt, height=100, disabled=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📄 Document Upload & Processing", "💬 Chat with Documents"])
    
    with tab1:
        st.header("Upload and Process PDF Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to process and add to your knowledge base"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Extract text from PDF
                        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                            extracted_text = st.session_state.processor.process_pdf_to_text(tmp_path)
                        
                        # Upload to search index
                        with st.spinner(f"Uploading {uploaded_file.name} to search index..."):
                            success = st.session_state.processor.upload_to_search_index(
                                extracted_text, uploaded_file.name
                            )
                        
                        if success:
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("All documents processed!")
                st.balloons()
    
    with tab2:
        st.header("Chat with Your Documents")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for doc in message["sources"]:
                            st.write(f"**{doc['source_filename']}** (Page {doc['page_number']}) - Score: {doc['score']:.4f}")
                            st.write(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
                            st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    try:
                        result = st.session_state.chatbot.chat(prompt)
                        response = result["response"]
                        sources = result["retrieved_documents"]
                        
                        st.markdown(response)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 View Sources"):
                                for doc in sources:
                                    st.write(f"**{doc['source_filename']}** (Page {doc['page_number']}) - Score: {doc['score']:.4f}")
                                    st.write(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
                                    st.divider()
                        
                    except Exception as e:
                        error_msg = f"เกิดข้อผิดพลาด: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 