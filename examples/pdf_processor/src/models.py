from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Callable, Awaitable, TypeAlias
from datetime import datetime
from enum import Enum
from pathlib import Path

# Define download status enum
class DownloadStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# Define paper models
class Author(BaseModel):
    name: str

class ArXivPaper(BaseModel):
    id: str
    title: str
    authors: List[Author]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: HttpUrl

class ArXivSearchResult(BaseModel):
    query: str
    total_results: int
    papers: List[ArXivPaper]

# Define PDF download models
class DownloadRequest(BaseModel):
    paper_id: str
    pdf_url: HttpUrl
    output_path: Path
    max_retries: int = 3
    timeout: int = 30  # seconds

class DownloadResult(BaseModel):
    paper_id: str
    pdf_url: HttpUrl
    output_path: Path
    status: DownloadStatus
    error_message: Optional[str] = None
    download_time: Optional[float] = None  # in seconds
    file_size: Optional[int] = None  # in bytes

# Define function type aliases
DownloaderFn: TypeAlias = Callable[[DownloadRequest], Awaitable[DownloadResult]]
BatchDownloaderFn: TypeAlias = Callable[[List[DownloadRequest]], Awaitable[List[DownloadResult]]]

# Define configuration
class DownloaderConfig(BaseModel):
    concurrent_downloads: int = 3
    rate_limit_delay: float = 3.0  # seconds between requests
    user_agent: str = "ArXivExplorer/1.0"
    timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 5.0  # seconds
    output_dir: Path = Field(default_factory=lambda: Path("tmp/pdfs"))
