import asyncio
import httpx
import json
import logging
import os
import time
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import backoff

from src.models import (
    DownloadRequest,
    DownloadResult,
    DownloadStatus,
    DownloaderConfig,
    BatchDownloaderFn,
    ArXivSearchResult
)

# Set up logging
logger = logging.getLogger(__name__)

def create_pdf_downloader(config: Optional[DownloaderConfig] = None) -> BatchDownloaderFn:
    """Factory function to create a PDF downloader that downloads PDFs from arXiv"""
    
    # Use default config if none provided
    if config is None:
        config = DownloaderConfig()
    
    # Ensure output directory exists (one-time setup)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a shared client session for all downloads
    client_params = {
        "timeout": httpx.Timeout(config.timeout),
        "follow_redirects": True,
        "headers": {
            "User-Agent": config.user_agent
        }
    }
    
    # Define backoff strategy for retries
    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.TimeoutException),
        max_tries=config.max_retries,
        factor=1,
        jitter=backoff.full_jitter
    )
    async def download_with_retry(client: httpx.AsyncClient, url: str, output_path: Path) -> Dict[str, Any]:
        """Download a file with retry logic"""
        start_time = time.time()
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Stream the file to disk to handle large files efficiently
            file_size = 0
            with open(output_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    file_size += len(chunk)
            
            download_time = time.time() - start_time
            return {
                "file_size": file_size,
                "download_time": download_time
            }
    
    def save_paper_record(paper_id: str, record_data: Dict[str, Any]) -> Path:
        """Save a paper record to its own JSON file"""
        record_path = config.output_dir / f"{paper_id}.json"
        with open(record_path, 'w') as f:
            json.dump(record_data, f, indent=2)
        logger.info(f"Saved paper record to {record_path}")
        return record_path
    
    def load_paper_record(paper_id: str) -> Optional[Dict[str, Any]]:
        """Load a paper record from its JSON file"""
        record_path = config.output_dir / f"{paper_id}.json"
        if record_path.exists():
            try:
                with open(record_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load paper record for {paper_id}: {e}")
        return None
    
    async def download_single_pdf(request: DownloadRequest, paper_metadata: Optional[Dict[str, Any]] = None) -> DownloadResult:
        """Inner function to download a single PDF"""
        logger.info(f"Downloading PDF for paper {request.paper_id} from {request.pdf_url}")
        
        # Check if already downloaded by looking for individual record file
        record = load_paper_record(request.paper_id)
        if record:
            logger.info(f"Paper {request.paper_id} already downloaded")
            output_path = Path(record["output_path"])
            
            # Verify the file still exists
            if output_path.exists():
                # Update metadata if new metadata is provided
                if paper_metadata and "metadata" in record:
                    # Merge new metadata with existing metadata, preferring new values
                    record["metadata"].update(paper_metadata)
                    # Save updated record
                    save_paper_record(request.paper_id, record)
                
                return DownloadResult(
                    paper_id=request.paper_id,
                    pdf_url=request.pdf_url,
                    output_path=output_path,
                    status=DownloadStatus.COMPLETED,
                    download_time=record.get("download_time", 0.0),
                    file_size=record.get("file_size", 0)
                )
            else:
                logger.warning(f"Previously downloaded file for {request.paper_id} not found, re-downloading")
        
        try:
            # Create a new client for each download to avoid session sharing issues
            async with httpx.AsyncClient(**client_params) as client:
                # Download the PDF with retry logic
                result = await download_with_retry(
                    client=client,
                    url=str(request.pdf_url),
                    output_path=request.output_path
                )
                
                # Get current datetime in ISO format
                current_datetime = datetime.datetime.now().isoformat()
                
                # Create record with enhanced metadata
                record = {
                    "paper_id": request.paper_id,
                    "pdf_url": str(request.pdf_url),
                    "output_path": str(request.output_path),
                    "download_time": result["download_time"],
                    "file_size": result["file_size"],
                    "download_date": current_datetime,
                    "metadata": paper_metadata or {},
                    # Full text will be added later in the pipeline
                    "full_text": None
                }
                
                # Save individual paper record
                save_paper_record(request.paper_id, record)
                
                return DownloadResult(
                    paper_id=request.paper_id,
                    pdf_url=request.pdf_url,
                    output_path=request.output_path,
                    status=DownloadStatus.COMPLETED,
                    download_time=result["download_time"],
                    file_size=result["file_size"]
                )
                
        except httpx.TimeoutException:
            error_message = f"Request timed out after {config.timeout} seconds"
            logger.error(f"Failed to download {request.paper_id}: {error_message}")
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=request.output_path,
                status=DownloadStatus.FAILED,
                error_message=error_message
            )
        except httpx.HTTPStatusError as e:
            error_message = f"HTTP error: {e.response.status_code} - {e.response.reason_phrase}"
            logger.error(f"Failed to download {request.paper_id}: {error_message}")
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=request.output_path,
                status=DownloadStatus.FAILED,
                error_message=error_message
            )
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to download {request.paper_id}: {error_message}", exc_info=True)
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=request.output_path,
                status=DownloadStatus.FAILED,
                error_message=error_message
            )
    
    async def batch_download_pdfs(requests: List[DownloadRequest], paper_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None) -> List[DownloadResult]:
        """Process a batch of download requests with rate limiting and concurrency control"""
        logger.info(f"Starting batch download of {len(requests)} PDFs")
        
        results = []
        semaphore = asyncio.Semaphore(config.concurrent_downloads)
        
        async def download_with_rate_limit(request: DownloadRequest) -> DownloadResult:
            async with semaphore:
                # Get metadata for this paper if available
                metadata = paper_metadata_map.get(request.paper_id) if paper_metadata_map else None
                result = await download_single_pdf(request, metadata)
                # Apply rate limiting to avoid overwhelming the server
                await asyncio.sleep(config.rate_limit_delay)
                return result
        
        # Create tasks for all downloads
        tasks = [download_with_rate_limit(request) for request in requests]
        
        # Wait for all downloads to complete
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                logger.info(f"Download {result.status} for {result.paper_id}")
            except Exception as e:
                logger.error(f"Unexpected error in download task: {str(e)}", exc_info=True)
                # We don't append a result here as we don't know which request failed
        
        # Count successes and failures
        successes = sum(1 for r in results if r.status == DownloadStatus.COMPLETED)
        failures = sum(1 for r in results if r.status == DownloadStatus.FAILED)
        
        logger.info(f"Batch download completed: {successes} successful, {failures} failed")
        
        return results
    
    # Return the batch download function with modified signature to accept metadata
    return batch_download_pdfs

def load_search_results_from_directory(input_dir: Path) -> List[ArXivSearchResult]:
    """Load all search results from JSON files in the input directory"""
    results = []
    
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist")
        return results
    
    for file_path in input_dir.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                search_result = ArXivSearchResult.model_validate(data)
                results.append(search_result)
                logger.info(f"Loaded search result from {file_path}: {search_result.query} with {len(search_result.papers)} papers")
        except Exception as e:
            logger.error(f"Failed to load search result from {file_path}: {str(e)}", exc_info=True)
    
    return results

def extract_paper_metadata(paper) -> Dict[str, Any]:
    """Extract metadata from an ArXiv paper object"""
    return {
        "title": paper.title,
        "authors": [author.model_dump() for author in paper.authors],
        "summary": paper.summary,
        "published": str(paper.published),
        "updated": str(paper.updated),
        "categories": paper.categories,
        # Additional fields can be added here as needed
    }

async def process_all_search_results(
    input_dir: Path, 
    output_dir: Path, 
    config: Optional[DownloaderConfig] = None
) -> Dict[str, List[DownloadResult]]:
    """Process all search results in the input directory and download PDFs"""
    if config is None:
        config = DownloaderConfig(output_dir=output_dir)
    else:
        config.output_dir = output_dir
    
    # Load all search results
    search_results = load_search_results_from_directory(input_dir)
    logger.info(f"Loaded {len(search_results)} search results")
    
    # Create downloader
    pdf_downloader = create_pdf_downloader(config)
    
    # Process each search result
    results_by_query = {}
    for search_result in search_results:
        # Create download requests and metadata map
        requests = []
        metadata_map = {}
        
        for paper in search_result.papers:
            # Create download request
            request = DownloadRequest(
                paper_id=paper.id,
                pdf_url=paper.pdf_url,
                output_path=output_dir / f"{paper.id}.pdf",
                max_retries=config.max_retries,
                timeout=config.timeout
            )
            requests.append(request)
            
            # Extract and store metadata
            metadata_map[paper.id] = extract_paper_metadata(paper)
            # Add query information to metadata
            metadata_map[paper.id]["query"] = search_result.query
        
        # Download PDFs with metadata
        logger.info(f"Processing search result: {search_result.query} with {len(requests)} papers")
        download_results = await pdf_downloader(requests, metadata_map)
        
        # Store results by query
        results_by_query[search_result.query] = download_results
    
    return results_by_query

async def main():
    """Main entry point for the PDF processor"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define directories
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "tmp" / "input_metadata"
    output_dir = base_dir / "tmp" / "pdfs"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure downloader
    config = DownloaderConfig(
        concurrent_downloads=5,  # Adjust based on your network and CPU
        rate_limit_delay=1.0,    # Be respectful to arXiv servers
        user_agent="ArXivExplorer/1.0 (https://github.com/cookieclicker123/ArXiv-Research-Agent/)",
        timeout=60, # PDFs can be large
        max_retries=3,
        retry_delay=5.0,
        output_dir=output_dir
    )
    
    # Process all search results
    results = await process_all_search_results(input_dir, output_dir, config)
    
    # Print summary
    total_papers = sum(len(r) for r in results.values())
    total_success = sum(sum(1 for d in r if d.status == DownloadStatus.COMPLETED) for r in results.values())
    total_failure = sum(sum(1 for d in r if d.status == DownloadStatus.FAILED) for r in results.values())
    
    logger.info(f"PDF processing complete: {total_papers} papers processed, {total_success} successful, {total_failure} failed")

if __name__ == "__main__":
    asyncio.run(main()) 