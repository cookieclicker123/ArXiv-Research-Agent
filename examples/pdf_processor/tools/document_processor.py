from typing import List, Dict, Any
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        
    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators"""
        # Base case: text is short enough
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end point for this chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # Try to find a natural break point
            best_end = end
            for sep in self.separators:
                # Look for separator in window around target length
                window_start = max(start, end - 100)
                window_end = min(len(text), end + 100)
                last_sep = text.rfind(sep, window_start, window_end)
                
                if last_sep != -1 and last_sep > start:
                    best_end = last_sep + len(sep)
                    break
            
            # Add chunk and move start point
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            start = best_end - self.chunk_overlap
        
        return chunks

def create_document_splitters():
    """Create dense and regular splitters"""
    dense_splitter = RecursiveTextSplitter(
        chunk_size=800,
        chunk_overlap=400
    )
    
    regular_splitter = RecursiveTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    return dense_splitter, regular_splitter

def process_document(paper_json_path: Path) -> List[Dict[str, Any]]:
    """Process a single document into chunks with metadata"""
    # Load the JSON file
    with open(paper_json_path, 'r') as f:
        paper_data = json.load(f)
    
    # Extract text
    text = paper_data.get('full_text', '')
    if not text:
        logger.warning(f"No full text found in {paper_json_path}")
        return []
    
    # Create splitters
    dense_splitter, regular_splitter = create_document_splitters()
    
    # For academic papers, we'll use regular chunking
    # but you could add specific detection for dense sections if needed
    splitter = regular_splitter
    chunks = splitter.split_text(text)
    
    paper_id = paper_data.get('paper_id', '')
    logger.info(f"Split document {paper_id} into {len(chunks)} chunks")
    
    # Create chunks with metadata
    doc_chunks = []
    for i, chunk_text in enumerate(chunks):
        # Copy all existing metadata from the paper
        chunk_data = {
            # Keep all original metadata
            **paper_data,
            # Add chunking metadata
            "chunk_id": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk_text),
            "chunking_strategy": "regular",
            # Replace full text with just this chunk's text
            "text": chunk_text
        }
        # Remove the original full_text to avoid duplication
        if "full_text" in chunk_data:
            del chunk_data["full_text"]
            
        doc_chunks.append(chunk_data)
    
    return doc_chunks 