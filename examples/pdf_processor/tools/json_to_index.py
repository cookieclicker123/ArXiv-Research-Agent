import json
import numpy as np
import faiss
import logging
import sys
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from document_processor import process_document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_and_split_texts(pdfs_folder: Path) -> List[Dict[str, Any]]:
    """Load JSON files and split into chunks"""
    chunks = []
    
    # Find all JSON files in the pdfs folder
    json_files = list(pdfs_folder.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {pdfs_folder}")
    
    for json_path in json_files:
        # Process document into chunks
        doc_chunks = process_document(json_path)
        chunks.extend(doc_chunks)
    
    logger.info(f"Created a total of {len(chunks)} chunks from all documents")
    return chunks

def create_faiss_index(
    pdfs_folder: Path,
    index_folder: Path,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> None:
    """Create and save FAISS index from document chunks"""
    # Load and split documents
    chunks = load_and_split_texts(pdfs_folder)
    
    if not chunks:
        logger.error("No chunks were created. Check if JSON files contain full_text field.")
        return
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    
    # Create embeddings
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Creating embeddings for {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Normalize vectors for cosine similarity
    logger.info("Normalizing vectors")
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # Get embedding dimension
    logger.info(f"Creating FAISS index with dimension {dimension}")
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
    index.add(embeddings)
    
    # Create output directory if it doesn't exist
    index_folder.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = index_folder / "faiss.index"
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, str(index_path))
    
    # Save chunks separately (FAISS only stores vectors)
    chunks_path = index_folder / "chunks.json"
    logger.info(f"Saving {len(chunks)} chunks to {chunks_path}")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)
    
    logger.info(f"Created FAISS index with {len(chunks)} chunks")
    logger.info(f"Index saved to {index_folder}")

def load_index(index_folder: Path):
    """Load saved index and chunks"""
    # Load FAISS index
    index_path = index_folder / "faiss.index"
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    
    # Load chunks
    chunks_path = index_folder / "chunks.json"
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    return index, chunks

def similarity_search(
    query: str,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    k: int = 4
):
    """Search for similar chunks"""
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Normalize query vector
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    
    # Return chunks with scores
    results = [
        (chunks[int(idx)], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0  # Ensure no negative indices
    ]
    
    return results

if __name__ == "__main__":
    # Define paths
    pdfs_folder = project_root / "tmp" / "pdfs"
    index_folder = project_root / "tmp" / "indices"
    
    # Create index
    create_faiss_index(pdfs_folder, index_folder) 