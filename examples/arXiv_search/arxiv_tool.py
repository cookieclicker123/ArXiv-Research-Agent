import logging
import arxiv
from pydantic import HttpUrl

from models import ArXivRequest, ArXivSearchResult, ArXivPaper, Author, ArXivFn

# Set up logging
logger = logging.getLogger(__name__)


def create_arxiv_client() -> ArXivFn:
    """Factory function to create an arXiv client function"""
    
    async def search_arxiv(request: ArXivRequest) -> ArXivSearchResult:
        """Search arXiv using the provided request parameters"""
        try:
            # Configure the search client
            client = arxiv.Client(
                page_size=request.max_results,
                delay_seconds=3.0,  # Be nice to the API
                num_retries=3
            )
            
            # Set up the search parameters
            search = arxiv.Search(
                query=request.query,
                max_results=request.max_results,
                sort_by=arxiv.SortCriterion.Relevance if request.sort_by == "relevance" else 
                       arxiv.SortCriterion.LastUpdatedDate if request.sort_by == "lastUpdatedDate" else
                       arxiv.SortCriterion.SubmittedDate
            )
            
            # Execute the search
            results = list(client.results(search))
            
            # Convert the results to our model format
            papers = []
            for result in results:
                # Extract authors
                authors = [Author(name=author.name) for author in result.authors]
                
                # Create the paper object
                paper = ArXivPaper(
                    id=result.get_short_id(),
                    title=result.title,
                    authors=authors,
                    summary=result.summary,
                    published=result.published,
                    updated=result.updated,
                    categories=result.categories,
                    pdf_url=HttpUrl(result.pdf_url)
                )
                papers.append(paper)
            
            # Create and return the search result
            return ArXivSearchResult(
                query=request.query,
                total_results=len(papers),
                papers=papers
            )
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            # Return empty result on error
            return ArXivSearchResult(
                query=request.query,
                total_results=0,
                papers=[]
            )
    
    return search_arxiv 