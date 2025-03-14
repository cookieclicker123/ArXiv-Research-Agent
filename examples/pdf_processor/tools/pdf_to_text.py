import os
import fitz  # PyMuPDF for reading PDFs
import json
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF"""
    try:
        start_time = time.time()
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        # Basic text cleaning
        text = text.replace('\n\n', ' ').replace('  ', ' ')
        
        extraction_time = time.time() - start_time
        logger.info(f"Extracted {len(text)} characters from {pdf_path} in {extraction_time:.2f} seconds")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def update_json_with_full_text(json_path, full_text):
    """Update a JSON file with the full text of a PDF"""
    try:
        # Load existing JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Update the full_text field
        data['full_text'] = full_text
        
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Updated {json_path} with full text")
        return True
    except Exception as e:
        logger.error(f"Error updating {json_path} with full text: {str(e)}")
        return False

def process_pdfs_in_directory(pdf_dir):
    """Process all PDFs in a directory and update corresponding JSON files"""
    pdf_dir = Path(pdf_dir)
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Process each PDF
    success_count = 0
    failure_count = 0
    
    for pdf_path in pdf_files:
        # Get paper ID from filename
        paper_id = pdf_path.stem
        
        # Check if corresponding JSON exists
        json_path = pdf_dir / f"{paper_id}.json"
        if not json_path.exists():
            logger.warning(f"No JSON file found for {paper_id}, skipping")
            continue
        
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_path)
        if full_text is None:
            logger.error(f"Failed to extract text from {pdf_path}, skipping")
            failure_count += 1
            continue
        
        # Update JSON with full text
        if update_json_with_full_text(json_path, full_text):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Processing complete: {success_count} successful, {failure_count} failed")
    return success_count, failure_count

if __name__ == "__main__":
    # Define the directory containing PDFs and JSON files
    pdf_dir = Path(__file__).parent.parent / "tmp" / "pdfs"
    
    # Process all PDFs in the directory
    process_pdfs_in_directory(pdf_dir) 