import sys
from pathlib import Path
import logging
import json
import faiss
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our models
from src.models import ArXivPaper, Author

def load_index(index_folder):
    """Load saved index and chunks"""
    # Load FAISS index
    index_path = Path(index_folder) / "faiss.index"
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    
    # Load chunks
    chunks_path = Path(index_folder) / "chunks.json"
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    return index, chunks

def similarity_search(
    query: str,
    index,
    chunks: List[Dict[str, Any]],
    model,
    k: int = 4,
    filters: Optional[Dict[str, Any]] = None
):
    """
    Search for similar chunks with optional filtering
    
    Args:
        query: The search query
        index: FAISS index
        chunks: List of chunk data
        model: SentenceTransformer model
        k: Number of results to return
        filters: Optional filters to apply to results
            - author: Filter by author name
            - paper_id: Filter by paper ID
            - title: Filter by paper title (substring match)
            - published_after: Filter by publication date (YYYY-MM-DD)
            - published_before: Filter by publication date (YYYY-MM-DD)
            - categories: Filter by paper categories (list of category codes)
    
    Returns:
        List of (chunk, score) tuples that match the query and filters
    """
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Normalize query vector
    faiss.normalize_L2(query_embedding)
    
    # Search - get more results than needed to allow for filtering
    search_k = k * 10 if filters else k
    scores, indices = index.search(query_embedding, min(search_k, len(chunks)))
    
    # Get initial results
    results = [
        (chunks[int(idx)], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0  # Ensure no negative indices
    ]
    
    # Apply filters if provided
    if filters:
        filtered_results = []
        for chunk, score in results:
            # Get metadata from chunk
            paper_id = chunk.get('paper_id', '')
            metadata = chunk.get('metadata', {})
            
            # Check all filters
            include = True
            
            # Author filter
            if 'author' in filters and include:
                author_filter = filters['author'].lower()
                authors = metadata.get('authors', [])
                author_names = [a.get('name', '').lower() for a in authors]
                include = any(author_filter in name for name in author_names)
            
            # Paper ID filter
            if 'paper_id' in filters and include:
                include = paper_id == filters['paper_id']
            
            # Title filter
            if 'title' in filters and include:
                title = metadata.get('title', '').lower()
                include = filters['title'].lower() in title
            
            # Date filters
            if 'published_after' in filters and include:
                published_str = metadata.get('published', '')
                if published_str:
                    try:
                        published_date = datetime.fromisoformat(published_str)
                        filter_date = datetime.fromisoformat(filters['published_after'])
                        include = published_date >= filter_date
                    except (ValueError, TypeError):
                        include = False
            
            if 'published_before' in filters and include:
                published_str = metadata.get('published', '')
                if published_str:
                    try:
                        published_date = datetime.fromisoformat(published_str)
                        filter_date = datetime.fromisoformat(filters['published_before'])
                        include = published_date <= filter_date
                    except (ValueError, TypeError):
                        include = False
            
            # Categories filter
            if 'categories' in filters and include:
                paper_categories = metadata.get('categories', [])
                filter_categories = filters['categories']
                include = any(cat in filter_categories for cat in paper_categories)
            
            # Add to filtered results if it passes all filters
            if include:
                filtered_results.append((chunk, score))
        
        results = filtered_results
    
    # Return top k results after filtering
    return results[:k]

def get_paper_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract paper metadata from a chunk into our model format"""
    paper_id = chunk.get('paper_id', '')
    metadata = chunk.get('metadata', {})
    
    # Convert to our model format
    return {
        'id': paper_id,
        'title': metadata.get('title', ''),
        'authors': metadata.get('authors', []),
        'summary': metadata.get('summary', ''),
        'published': metadata.get('published', ''),
        'updated': metadata.get('updated', ''),
        'categories': metadata.get('categories', []),
        'pdf_url': chunk.get('pdf_url', f"http://arxiv.org/pdf/{paper_id}")
    }

def print_search_results(query: str, results, filters: Optional[Dict[str, Any]] = None):
    """Print search results in a readable format"""
    print(f"\n{'-'*80}")
    print(f"Results for query: '{query}'")
    if filters:
        print(f"Filters: {filters}")
    print(f"{'-'*80}")
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {i} (similarity score: {score:.3f}):")
        
        # Print metadata
        paper_id = chunk.get('paper_id', 'Unknown')
        metadata = chunk.get('metadata', {})
        title = metadata.get('title', 'Unknown title')
        authors = metadata.get('authors', [])
        author_names = [a.get('name', '') for a in authors]
        
        print(f"Paper: {paper_id}")
        print(f"Title: {title}")
        print(f"Authors: {', '.join(author_names)}")
        print(f"Published: {metadata.get('published', 'Unknown')}")
        print(f"Categories: {', '.join(metadata.get('categories', []))}")
        print(f"Chunk: {chunk.get('chunk_id', '?')}/{chunk.get('total_chunks', '?')}")
        
        # Print text snippet
        text = chunk.get('text', '')
        print(f"\nText snippet: {text[:700]}...")
        
        print(f"{'-'*40}")

def run_test_queries():
    """Run a set of test queries against the index"""
    # Load index and chunks
    index_folder = project_root / "tmp" / "indices"
    index, chunks = load_index(index_folder)
    
    # Load model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Test queries with various filters
    test_cases = [
        {
            "query": "Explain in detail what a quantum circuit model is",
            "filters": None
        },
        {
            "query": "What is the Forward-Forward algorithm?",
            "filters": None
        },
        {
            "query": "Neural networks",
            "filters": {"author": "Hinton"}
        },
        {
            "query": "Quantum computing",
            "filters": {"categories": ["quant-ph"]}
        }
    ]
    
    for case in test_cases:
        query = case["query"]
        filters = case["filters"]
        
        # Get search results
        results = similarity_search(
            query=query,
            index=index,
            chunks=chunks,
            model=model,
            k=3,
            filters=filters
        )
        
        # Print results
        print_search_results(query, results, filters)
        
        print("\nPress Enter to continue to next query...")
        input()

def search_papers(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 3
) -> List[Dict[str, Union[ArXivPaper, List[Dict[str, Any]]]]]:
    """
    Search for papers matching the query and filters
    
    Returns a list of papers with their matching chunks
    """
    # Load index and chunks
    index_folder = project_root / "tmp" / "indices"
    index, chunks = load_index(index_folder)
    
    # Load model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Search
    results = similarity_search(
        query=query,
        index=index,
        chunks=chunks,
        model=model,
        k=k,
        filters=filters
    )
    
    # Group by paper
    papers = {}
    for chunk, score in results:
        paper_id = chunk.get('paper_id', '')
        if paper_id not in papers:
            # Extract paper metadata
            metadata = get_paper_metadata(chunk)
            
            # Create paper entry
            papers[paper_id] = {
                'paper': ArXivPaper(**metadata),
                'chunks': []
            }
        
        # Add chunk to paper
        papers[paper_id]['chunks'].append({
            'text': chunk.get('text', ''),
            'score': score,
            'chunk_id': chunk.get('chunk_id', 0)
        })
    
    # Convert to list and sort by best chunk score
    paper_list = list(papers.values())
    paper_list.sort(key=lambda p: max(c['score'] for c in p['chunks']), reverse=True)
    
    return paper_list

if __name__ == "__main__":
    run_test_queries()