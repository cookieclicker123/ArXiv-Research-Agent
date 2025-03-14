import asyncio
import logging
import random
import json
from typing import List, Optional

from src.models import (
    DownloadRequest,
    DownloadResult,
    DownloadStatus,
    DownloaderConfig,
    BatchDownloaderFn,
)

# Set up logging
logger = logging.getLogger(__name__)

def create_mock_downloader(config: Optional[DownloaderConfig] = None) -> BatchDownloaderFn:
    """Factory function to create a mock PDF downloader that simulates downloading PDFs"""
    
    # Use default config if none provided
    if config is None:
        config = DownloaderConfig()
    
    # Ensure output directory exists (one-time setup)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load any existing download records to avoid re-downloading
    download_record_path = config.output_dir / "download_records.json"
    download_records = {}
    if download_record_path.exists():
        try:
            with open(download_record_path, 'r') as f:
                download_records = json.load(f)
            logger.info(f"Loaded {len(download_records)} existing download records")
        except Exception as e:
            logger.warning(f"Failed to load download records: {e}")
    
    async def download_single_pdf(request: DownloadRequest) -> DownloadResult:
        """Inner function to simulate downloading a single PDF"""
        logger.info(f"Mock downloading PDF for paper {request.paper_id} from {request.pdf_url}")
        
        # Check if already downloaded
        if request.paper_id in download_records:
            logger.info(f"Paper {request.paper_id} already downloaded")
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=request.output_path,
                status=DownloadStatus.COMPLETED,
                download_time=download_records[request.paper_id].get("download_time", 0.0),
                file_size=download_records[request.paper_id].get("file_size", 0)
            )
        
        # Simulate download time
        download_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(download_time)
        
        # Simulate success/failure (90% success rate)
        success = random.random() < 0.9
        
        if success:
            # Simulate file creation
            output_path = request.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a small mock PDF file
            file_size = random.randint(500_000, 5_000_000)  # 500KB to 5MB
            with open(output_path, 'wb') as f:
                f.write(b'%PDF-1.5\n' + b'x' * 100 + b'\n%%EOF\n')
            
            # Record the download
            download_records[request.paper_id] = {
                "pdf_url": str(request.pdf_url),
                "output_path": str(request.output_path),
                "download_time": download_time,
                "file_size": file_size,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Save updated records
            with open(download_record_path, 'w') as f:
                json.dump(download_records, f, indent=2)
            
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=output_path,
                status=DownloadStatus.COMPLETED,
                download_time=download_time,
                file_size=file_size
            )
        else:
            # Simulate failure
            error_messages = [
                "Connection timeout",
                "Server error (503)",
                "SSL certificate verification failed",
                "Too many redirects",
                "File not found (404)"
            ]
            error_message = random.choice(error_messages)
            
            return DownloadResult(
                paper_id=request.paper_id,
                pdf_url=request.pdf_url,
                output_path=request.output_path,
                status=DownloadStatus.FAILED,
                error_message=error_message,
                download_time=download_time
            )
    
    async def batch_download_pdfs(requests: List[DownloadRequest]) -> List[DownloadResult]:
        """Process a batch of download requests with rate limiting"""
        logger.info(f"Starting batch download of {len(requests)} PDFs")
        
        results = []
        semaphore = asyncio.Semaphore(config.concurrent_downloads)
        
        async def download_with_rate_limit(request: DownloadRequest) -> DownloadResult:
            async with semaphore:
                result = await download_single_pdf(request)
                # Simulate rate limiting
                await asyncio.sleep(config.rate_limit_delay)
                return result
        
        # Create tasks for all downloads
        tasks = [download_with_rate_limit(request) for request in requests]
        
        # Wait for all downloads to complete
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            logger.info(f"Download {result.status} for {result.paper_id}")
        
        # Count successes and failures
        successes = sum(1 for r in results if r.status == DownloadStatus.COMPLETED)
        failures = sum(1 for r in results if r.status == DownloadStatus.FAILED)
        
        logger.info(f"Batch download completed: {successes} successful, {failures} failed")
        
        return results
    
    # Return the batch download function
    return batch_download_pdfs