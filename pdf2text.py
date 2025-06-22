import os
import base64
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI


# Configuration
SRC_DIRECTORY = 'data'
DST_DIRECTORY = 'data_txt'
IMAGES_DIRECTORY = 'images'
IMAGE_DPI = 300
MODEL_NAME = "gpt-4.1-mini"
API_VERSION = "2024-02-15-preview"
MAX_TOKENS = 4096

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_azure_client() -> AzureOpenAI:
    """Set up and return Azure OpenAI client."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        raise ValueError(
            "Missing environment variables: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
    
    return AzureOpenAI(
        api_key=api_key,
        api_version=API_VERSION,
        azure_endpoint=endpoint
    )


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    Path(DST_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_DIRECTORY).mkdir(parents=True, exist_ok=True)


def find_all_pdf_files() -> List[Path]:
    """Find all PDF files in the source directory."""
    src_path = Path(SRC_DIRECTORY)
    
    if not src_path.exists():
        logger.warning(f"Source directory {SRC_DIRECTORY} does not exist")
        return []
    
    pdf_files = list(src_path.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def convert_pdf_to_images(pdf_path: Path) -> List[bytes]:
    """Convert PDF pages to PNG images and save them."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        pdf_name = pdf_path.stem
        
        # Calculate zoom for desired DPI
        zoom = IMAGE_DPI / 72  # 72 is default DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=matrix)
            img_data = pixmap.tobytes("png")
            images.append(img_data)
            
            # Save image to disk
            image_filename = f"{pdf_name}_page_{page_num + 1}.png"
            image_path = Path(IMAGES_DIRECTORY) / image_filename
            
            with open(image_path, 'wb') as img_file:
                img_file.write(img_data)
            
            logger.debug(f"Saved image: {image_filename}")
        
        doc.close()
        return images
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise


def extract_text_from_image(image_data: bytes, client: AzureOpenAI) -> str:
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
        response = client.chat.completions.create(
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
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content or ""
        
    except Exception as e:
        error_msg = f"Error extracting text from image: {e}"
        logger.error(error_msg)
        return error_msg


def get_output_file_path(pdf_path: Path) -> Path:
    """Get the output text file path for a given PDF file."""
    # Keep the same directory structure in output
    src_path = Path(SRC_DIRECTORY)
    relative_path = pdf_path.relative_to(src_path)
    txt_filename = relative_path.with_suffix('.txt')
    
    output_path = Path(DST_DIRECTORY) / txt_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path


def process_single_pdf(pdf_path: Path, client: AzureOpenAI) -> bool:
    """Process a single PDF file and extract text."""
    output_path = get_output_file_path(pdf_path)
    logger.info(f"Processing {pdf_path} -> {output_path}")
    
    try:
        # Step 1: Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            logger.warning(f"No images extracted from {pdf_path}")
            return False
        
        # Step 2: Extract text from each image
        extracted_pages = []
        for i, image_data in enumerate(images, 1):
            logger.info(f"  Extracting text from page {i}/{len(images)}...")
            text = extract_text_from_image(image_data, client)
            extracted_pages.append(f"\n--- Page {i} ---\n{text}")
        
        # Step 3: Save all extracted text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(extracted_pages))
        
        logger.info(f"  Successfully saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return False


def main():
    """Main function to process all PDF files."""
    try:
        # Setup
        logger.info("Starting PDF to text conversion...")
        create_directories()
        client = setup_azure_client()
        
        # Find all PDF files
        pdf_files = find_all_pdf_files()
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return
        
        # Process each PDF file
        successful = 0
        failed = 0
        
        for pdf_path in pdf_files:
            if process_single_pdf(pdf_path, client):
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info(f"Processing complete!")
        logger.info(f"Successfully processed: {successful} files")
        if failed > 0:
            logger.warning(f"Failed to process: {failed} files")
        else:
            logger.info("All files processed successfully!")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
