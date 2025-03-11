from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union
from datetime import datetime
from enum import Enum


# Groq LLM Models
class GroqMessage(BaseModel):
    """A message in a Groq chat conversation"""
    role: str
    content: str

class GroqChatRequest(BaseModel):
    """Request to the Groq chat API"""
    model: str
    messages: List[GroqMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = None

class GroqChatResponse(BaseModel):
    """Response from the Groq chat API"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Union[int, float]]  # Allow both int and float values

# Type alias for Groq LLM function
GroqLLMFn = Callable[[GroqChatRequest], Awaitable[GroqChatResponse]]


# arXiv Models
class ArXivSearchField(str, Enum):
    """Available search fields in the arXiv API"""
    TITLE = "ti"
    AUTHOR = "au"
    ABSTRACT = "abs"
    CATEGORY = "cat"
    SUBMITTED_DATE = "submittedDate"
    
    @classmethod
    def description(cls) -> Dict[str, str]:
        return {
            cls.TITLE.value: "Title search",
            cls.AUTHOR.value: "Author search",
            cls.ABSTRACT.value: "Abstract search",
            cls.CATEGORY.value: "Category search",
            cls.SUBMITTED_DATE.value: "Date range in format [YYYYMMDD000000 TO YYYYMMDD235959]"
        }

class ArXivCategory(str, Enum):
    """Common arXiv categories"""
    CS_AI = "cs.AI"
    CS_CL = "cs.CL"
    CS_CV = "cs.CV"
    CS_LG = "cs.LG"
    QUANT_PH = "quant-ph"
    COND_MAT = "cond-mat"
    PHYSICS = "physics"
    MATH = "math"
    STAT = "stat"
    
    @classmethod
    def description(cls) -> Dict[str, str]:
        return {
            cls.CS_AI.value: "Artificial Intelligence",
            cls.CS_CL.value: "Computation and Language",
            cls.CS_CV.value: "Computer Vision",
            cls.CS_LG.value: "Machine Learning",
            cls.QUANT_PH.value: "Quantum Physics",
            cls.COND_MAT.value: "Condensed Matter",
            cls.PHYSICS.value: "Physics (general)",
            cls.MATH.value: "Mathematics",
            cls.STAT.value: "Statistics"
        }

class Author(BaseModel):
    """Author of an arXiv paper"""
    name: str

class ArXivRequest(BaseModel):
    """Request to the arXiv API"""
    query: str
    max_results: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"

class DateRange(BaseModel):
    start_date: Optional[datetime] = Field(default=None, description="Start date for range")
    end_date: Optional[datetime] = Field(default=None, description="End date for range")

class ArXivPaper(BaseModel):
    """An arXiv paper"""
    id: str
    title: str
    authors: List[Author]
    summary: str
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: HttpUrl

class ArXivSearchResult(BaseModel):
    """Result from an arXiv search"""
    query: str
    total_results: int
    papers: List[ArXivPaper]

# Type alias for arXiv search function
ArXivFn = Callable[[ArXivRequest], Awaitable[ArXivSearchResult]]