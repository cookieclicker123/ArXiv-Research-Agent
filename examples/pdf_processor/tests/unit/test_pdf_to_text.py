import pytest
import json
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.pdf_to_text import extract_text_from_pdf, update_json_with_full_text, process_pdfs_in_directory

# Test directories
PDF_DIR = project_root / "tmp" / "pdfs"

def test_extract_text_from_pdf():
    """Test extracting text from a PDF file"""
    # Find a PDF file to test with
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found for testing")
    
    pdf_path = pdf_files[0]
    text = extract_text_from_pdf(pdf_path)
    
    assert text is not None
    assert len(text) > 0
    assert isinstance(text, str)
    
    # Check for common PDF content markers
    assert "abstract" in text.lower() or "introduction" in text.lower() or "references" in text.lower()

def test_update_json_with_full_text():
    """Test updating a JSON file with full text"""
    # Find a JSON file to test with
    json_files = list(PDF_DIR.glob("*.json"))
    if not json_files:
        pytest.skip("No JSON files found for testing")
    
    json_path = json_files[0]
    
    # Create a backup of the original JSON
    backup_path = json_path.with_suffix(".json.bak")
    with open(json_path, 'r') as f:
        original_data = json.load(f)
    with open(backup_path, 'w') as f:
        json.dump(original_data, f)
    
    try:
        # Test updating the JSON
        sample_text = "This is a test full text content."
        result = update_json_with_full_text(json_path, sample_text)
        
        assert result is True
        
        # Verify the JSON was updated
        with open(json_path, 'r') as f:
            updated_data = json.load(f)
        
        assert 'full_text' in updated_data
        assert updated_data['full_text'] == sample_text
    finally:
        # Restore the original JSON
        with open(backup_path, 'r') as f:
            original_data = json.load(f)
        with open(json_path, 'w') as f:
            json.dump(original_data, f, indent=2)
        
        # Remove backup
        backup_path.unlink()

def test_process_single_pdf():
    """Test processing a single PDF and updating its JSON"""
    # Find a PDF and its corresponding JSON
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found for testing")
    
    for pdf_path in pdf_files:
        json_path = pdf_path.with_suffix(".json")
        if json_path.exists():
            break
    else:
        pytest.skip("No matching PDF and JSON files found for testing")
    
    # Create a backup of the original JSON
    backup_path = json_path.with_suffix(".json.bak")
    with open(json_path, 'r') as f:
        original_data = json.load(f)
    with open(backup_path, 'w') as f:
        json.dump(original_data, f)
    
    try:
        # Process the PDF
        paper_id = pdf_path.stem
        text = extract_text_from_pdf(pdf_path)
        result = update_json_with_full_text(json_path, text)
        
        assert result is True
        
        # Verify the JSON was updated
        with open(json_path, 'r') as f:
            updated_data = json.load(f)
        
        assert 'full_text' in updated_data
        assert updated_data['full_text'] is not None
        assert len(updated_data['full_text']) > 0
        
        # Check that the text contains meaningful content
        assert len(updated_data['full_text']) > 100  # Arbitrary minimum length
    finally:
        # Restore the original JSON
        with open(backup_path, 'r') as f:
            original_data = json.load(f)
        with open(json_path, 'w') as f:
            json.dump(original_data, f, indent=2)
        
        # Remove backup
        backup_path.unlink()