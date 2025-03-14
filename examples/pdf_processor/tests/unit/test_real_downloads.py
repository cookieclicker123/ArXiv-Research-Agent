import pytest
import json
import sys
from pathlib import Path
import logging

# Add the project root to the Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import (
    DownloadRequest,
    DownloadStatus,
    DownloaderConfig,
    ArXivSearchResult
)
from src.pdf_downloader import create_pdf_downloader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test directories
BASE_DIR = project_root
INPUT_DIR = BASE_DIR / "tmp" / "input_metadata"
OUTPUT_DIR = BASE_DIR / "tmp" / "pdfs"

@pytest.fixture(scope="module")
def setup_dirs():
    """Set up directories for testing"""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not INPUT_DIR.exists():
        pytest.skip(f"Input directory {INPUT_DIR} does not exist. Skipping real download tests.")
    
    # Check if there are JSON files in the input directory
    json_files = list(INPUT_DIR.glob("*.json"))
    if not json_files:
        pytest.skip(f"No JSON files found in {INPUT_DIR}. Skipping real download tests.")
    
    return json_files

def load_papers_from_json_files(json_files, max_papers_per_file=1):
    """Load the first paper from each JSON file"""
    papers = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Try to parse as ArXivSearchResult
            search_result = ArXivSearchResult.model_validate(data)
            
            # Take only the first paper from each file to avoid excessive downloads
            if search_result.papers:
                for paper in search_result.papers[:max_papers_per_file]:
                    papers.append({
                        'paper_id': paper.id,
                        'pdf_url': paper.pdf_url,
                        'title': paper.title,
                        'source_file': json_file.name
                    })
                    
            logger.info(f"Loaded {min(max_papers_per_file, len(search_result.papers))} papers from {json_file}")
                
        except Exception as e:
            logger.error(f"Error loading {json_file}: {str(e)}")
    
    return papers

@pytest.mark.real
@pytest.mark.asyncio
async def test_real_pdf_downloads(setup_dirs):
    """Test downloading real PDFs from arXiv using actual JSON input files"""
    json_files = setup_dirs
    
    # Load papers from JSON files (first paper from each file)
    papers = load_papers_from_json_files(json_files, max_papers_per_file=1)
    
    if not papers:
        pytest.skip("No valid papers found in JSON files. Skipping test.")
    
    logger.info(f"Found {len(papers)} papers to download")
    
    # Create download requests
    requests = [
        DownloadRequest(
            paper_id=paper['paper_id'],
            pdf_url=paper['pdf_url'],
            output_path=OUTPUT_DIR / f"{paper['paper_id']}.pdf",
            max_retries=3,
            timeout=60
        )
        for paper in papers
    ]
    
    # Configure downloader
    config = DownloaderConfig(
        concurrent_downloads=2,  # Limit concurrency to be respectful
        rate_limit_delay=1.0,    # Add delay between requests
        user_agent="ArXivExplorer/1.0 (Testing)",
        output_dir=OUTPUT_DIR
    )
    
    # Create downloader
    pdf_downloader = create_pdf_downloader(config)
    
    # Download PDFs
    logger.info(f"Starting download of {len(requests)} PDFs")
    results = await pdf_downloader(requests)
    
    # Check results
    assert len(results) == len(requests)
    
    # Count successes and failures
    successes = [r for r in results if r.status == DownloadStatus.COMPLETED]
    failures = [r for r in results if r.status == DownloadStatus.FAILED]
    
    logger.info(f"Downloads completed: {len(successes)} successful, {len(failures)} failed")
    
    # Log details of failures
    for failure in failures:
        logger.error(f"Failed to download {failure.paper_id}: {failure.error_message}")
    
    # Verify at least some downloads were successful
    assert len(successes) > 0, "No PDFs were successfully downloaded"
    
    # Verify files exist and are valid PDFs
    for result in successes:
        assert result.output_path.exists(), f"Output file {result.output_path} does not exist"
        assert result.file_size > 0, f"Output file {result.output_path} is empty"
        
        # Check if it's a valid PDF
        with open(result.output_path, 'rb') as f:
            content = f.read(10)  # Read first 10 bytes
            assert content.startswith(b'%PDF'), f"File {result.output_path} is not a valid PDF"
        
        logger.info(f"Successfully downloaded {result.paper_id} to {result.output_path}")
    
    # Print summary
    for i, paper in enumerate(papers):
        status = "✅ Success" if results[i].status == DownloadStatus.COMPLETED else "❌ Failed"
        logger.info(f"{status}: {paper['paper_id']} - {paper['title']} (from {paper['source_file']})")
