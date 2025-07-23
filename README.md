# PDF RAG Chat System 📚

A Streamlit web application that allows you to upload PDF documents, process them using Azure OpenAI Vision API, store them in Azure AI Search, and chat with your documents using RAG (Retrieval-Augmented Generation).

## Features

- **📄 PDF Upload & Processing**: Upload multiple PDF files and extract text using Azure OpenAI Vision API
- **🔍 Vector Search**: Store processed documents in Azure AI Search with vector embeddings
- **💬 RAG Chat Interface**: Chat with your documents using retrieval-augmented generation
- **📚 Source Attribution**: View source documents and page numbers for each response
- **🌐 Web Interface**: Easy-to-use Streamlit web interface

## Prerequisites

- Python 3.12+
- Azure OpenAI Service with GPT-4 Vision and text-embedding-3-large models
- Azure AI Search service
- Required Python packages (see requirements.txt)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with your Azure credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-azure-openai-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=https://your-openai-resource.openai.azure.com/

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key
AZURE_SEARCH_INDEX_NAME=your-index-name
```

*You can use `env_example.txt` as a template.*

### 3. Set Up Azure AI Search Index

Before using the application, make sure your Azure AI Search index is properly configured with the following fields:

- `id` (Edm.String, key: true)
- `content` (Edm.String, searchable: true)
- `content_vector` (Collection(Edm.Single), searchable: true, retrievable: true, dimensions: 3072)
- `source_filename` (Edm.String, searchable: true, retrievable: true)
- `page_number` (Edm.Int32, retrievable: true)
- `last_modified` (Edm.DateTimeOffset, retrievable: true)

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## How to Use

### Document Upload & Processing

1. Navigate to the **"Document Upload & Processing"** tab
2. Click **"Choose PDF files"** to select one or more PDF documents
3. Click **"Process Documents"** to:
   - Extract text from each PDF page using Azure OpenAI Vision API
   - Split the text into chunks
   - Generate embeddings for each chunk
   - Upload to Azure AI Search index

### Chat with Documents

1. Navigate to the **"Chat with Documents"** tab
2. Type your question in the chat input
3. The system will:
   - Search for relevant document chunks using vector similarity
   - Generate a response using the retrieved context
   - Display source documents with page numbers

### Features

- **Progress Tracking**: Real-time progress updates during document processing
- **Error Handling**: Clear error messages and graceful failure handling
- **Source Attribution**: View which documents and pages were used for each response
- **Chat History**: Persistent chat history within the session
- **Clear Chat**: Reset the conversation at any time

## Architecture

The application consists of several key components:

1. **DocumentProcessor**: Handles PDF-to-text conversion and Azure AI Search indexing
2. **RAGChatbot**: Manages document retrieval and response generation
3. **Streamlit UI**: Provides the web interface for user interaction

### Document Processing Pipeline

1. **PDF Upload** → Convert PDF pages to images
2. **Text Extraction** → Use Azure OpenAI Vision API to extract text from images
3. **Text Chunking** → Split text into manageable chunks with overlap
4. **Embedding Generation** → Create vector embeddings for each chunk
5. **Index Upload** → Store chunks with metadata in Azure AI Search

### RAG Pipeline

1. **Query Processing** → Convert user question to vector embedding
2. **Document Retrieval** → Find relevant chunks using vector similarity search
3. **Context Assembly** → Combine retrieved chunks into context
4. **Response Generation** → Use Azure OpenAI to generate response with context
5. **Source Attribution** → Display source documents and page numbers

## Configuration

You can modify the following parameters in `app.py`:

- `IMAGE_DPI`: Resolution for PDF-to-image conversion (default: 300)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `BATCH_SIZE`: Number of documents to upload per batch (default: 10)
- `MAX_TOKENS`: Maximum tokens for text extraction (default: 4096)

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure all required variables are set in `.env`
2. **Azure Service Errors**: Check your Azure service quotas and API keys
3. **Large PDF Files**: Processing may take time for large documents
4. **Memory Issues**: Large batches may cause memory issues; reduce `BATCH_SIZE`

### Error Messages

- `"Missing environment variables"`: Check your `.env` file configuration
- `"Error initializing services"`: Verify your Azure credentials and service availability
- `"No relevant documents found"`: Upload and process documents before chatting

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please check the troubleshooting section or create an issue in the project repository. 