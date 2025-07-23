import os
import re
import unicodedata
import datetime
import gc
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

# Azure AI Search Configuration
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

BATCH_SIZE = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(char for char in text if char.isprintable())
    text = text.strip()
    return text

def get_embedding(text: str, client: AzureOpenAI):
    cleaned_text = clean_text(text)
    if len(cleaned_text.strip()) < 10:
        return None
    embedding_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=EMBEDDING_ENDPOINT
    )
    response = embedding_client.embeddings.create(
        input=cleaned_text,
        model=EMBEDDING_DEPLOYMENT,
        encoding_format="float"
    )
    return response.data[0].embedding

def sanitize_document_id(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    sanitized = re.sub(r'[^a-zA-Z0-9-]', '_', base_name)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = 'doc_' + sanitized
    return sanitized

def get_all_txt_files(root_dir):
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
    return txt_files

def split_by_recursive_chunking(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text using LangChain's RecursiveCharacterTextSplitter.
    This splits by paragraphs first, then sentences, then words.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def split_by_page(text):
    # Split on --- Page N --- markers, keep the marker with the chunk
    pattern = r'(--- Page (\d+) ---)'
    splits = re.split(pattern, text)
    chunks = []
    page_numbers = []
    i = 1
    while i < len(splits):
        marker = splits[i]
        page_number = splits[i+1]
        content = splits[i+2] if (i+2) < len(splits) else ''
        chunk_text = f'{marker}{content}'
        chunks.append(chunk_text.strip())
        page_numbers.append(int(page_number))
        i += 3
    return chunks, page_numbers

def main():
    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
    src_root = 'data_txt'
    txt_files = get_all_txt_files(src_root)
    print(f"Found {len(txt_files)} .txt files.")
    for txt_path in txt_files:
        print(f"Processing: {txt_path}")  # Log which document is being processed
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks, page_numbers = split_by_recursive_chunking(text)
        document_id = os.path.basename(txt_path)
        sanitized_doc_id = sanitize_document_id(document_id)
        source_filename = document_id
        source_filename_en = document_id
        rel_dir = os.path.dirname(os.path.relpath(txt_path, src_root))
        # Remove src_root from path_field
        if rel_dir:
            path_field = rel_dir + '/'
        else:
            path_field = ''
        docs_to_upload = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk, openai_client)
            except Exception as e:
                print(f"Error getting embedding for chunk {i} in {txt_path}: {e}")
                continue
            if embedding is None:
                continue
            doc = {
                "id": f"{sanitized_doc_id}_{i}",
                "content": clean_text(chunk),
                "content_vector": embedding,
                "source_filename": source_filename[:-4] + ".pdf",
                "last_modified": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "page_number": page_numbers[i],
            }
            docs_to_upload.append(doc)
            if len(docs_to_upload) >= BATCH_SIZE:
                search_client.upload_documents(docs_to_upload)
                print(f"Uploaded {len(docs_to_upload)} docs from {txt_path}")
                docs_to_upload = []
                gc.collect()
        if docs_to_upload:
            search_client.upload_documents(docs_to_upload)
            print(f"Uploaded {len(docs_to_upload)} docs from {txt_path}")
            gc.collect()
    print("All .txt files processed.")
if __name__ == "__main__":
    main()
