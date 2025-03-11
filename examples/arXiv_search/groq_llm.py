import os
import json
import logging
import datetime
import traceback
import re
from pathlib import Path
import inspect

from models import GroqChatRequest, GroqMessage, ArXivRequest, ArXivSearchResult, ArXivSearchField, ArXivCategory, GroqLLMFn
from groq import create_groq_client
from arxiv_tool import create_arxiv_client
from prompts import CONVERSION_PROMPT

# Set up logging
logger = logging.getLogger(__name__)

# Ensure results directory exists
RESULTS_DIR = Path("arXiv_results")
RESULTS_DIR.mkdir(exist_ok=True)

def create_groq_llm() -> GroqLLMFn:
    """Factory function to create a Groq LLM for arXiv search"""
    
    # Create the clients
    groq_client = create_groq_client()
    arxiv_client = create_arxiv_client()
    
    # Get model name from environment or use default
    model_name = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
    
    async def query_arxiv(user_query: str) -> ArXivSearchResult:
        """Process natural language query through Groq and arXiv"""
        try:
            # Step 1: Convert natural language to arXiv syntax using Groq
            logger.info(f"Converting query: {user_query}")
            
            # Debug the description methods
            try:
                logger.debug(f"ArXivSearchField type: {type(ArXivSearchField)}")
                logger.debug(f"ArXivSearchField dir: {dir(ArXivSearchField)}")
                
                # Check if description is a method or attribute
                if hasattr(ArXivSearchField, 'description'):
                    logger.debug(f"description is: {inspect.getattr_static(ArXivSearchField, 'description')}")
                    
                # Try to call the description method
                search_field_desc = None
                category_desc = None
                
                try:
                    search_field_desc = ArXivSearchField.description()
                    logger.debug(f"ArXivSearchField.description() result: {search_field_desc}")
                except Exception as desc_error:
                    logger.error(f"Error calling ArXivSearchField.description(): {desc_error}")
                    logger.error(traceback.format_exc())
                
                try:
                    category_desc = ArXivCategory.description()
                    logger.debug(f"ArXivCategory.description() result: {category_desc}")
                except Exception as desc_error:
                    logger.error(f"Error calling ArXivCategory.description(): {desc_error}")
                    logger.error(traceback.format_exc())
                
                # Fallback descriptions if methods fail
                if search_field_desc is None:
                    search_field_desc = {
                        "ti": "Title search",
                        "au": "Author search",
                        "abs": "Abstract search",
                        "cat": "Category search",
                        "submittedDate": "Date range in format [YYYYMMDD000000 TO YYYYMMDD235959]"
                    }
                
                if category_desc is None:
                    category_desc = {
                        "cs.AI": "Artificial Intelligence",
                        "quant-ph": "Quantum Physics",
                        "physics": "Physics",
                        "math": "Mathematics",
                        "cs.LG": "Machine Learning"
                    }
                
                # Format the conversion prompt with our models
                formatted_prompt = CONVERSION_PROMPT.format(
                    ArXivSearchField=str(ArXivSearchField.__members__),
                    ArXivSearchField_description=str(search_field_desc),
                    ArXivCategory=str(ArXivCategory.__members__),
                    ArXivCategory_description=str(category_desc),
                    ArXivRequest=str(ArXivRequest.model_json_schema())
                )
            except Exception as format_error:
                logger.error(f"Error formatting prompt: {format_error}")
                logger.error(traceback.format_exc())
                # Use a more comprehensive fallback prompt based on the original
                formatted_prompt = """
                You are an expert at converting natural language queries into arXiv API search syntax.
                
                arXiv search syntax uses:
                - Boolean operators: AND, OR, NOT (must be uppercase)
                - Field-specific search: ti (title), au (author), abs (abstract), cat (category)
                - Parentheses for grouping terms: (quantum AND computing)
                - Exact phrase matching with quotes: "quantum computing"
                - Wildcards: quantum* matches quantum, quantum mechanics, etc.
                
                For date filtering, use:
                - submittedDate:[YYYYMMDD000000 TO YYYYMMDD235959]
                  Example: submittedDate:[20230101000000 TO 20250331235959]
                
                Here are some examples of the conversion:
                1. "Find recent papers on quantum computing" → "cat:quant-ph AND (quantum computing) AND submittedDate:[20230101000000 TO 20250331235959]"
                2. "Papers by John Smith about neural networks" → "au:Smith_J AND (neural networks)"
                3. "Latest research on climate change in physics" → "cat:physics AND (climate change) AND submittedDate:[20230101000000 TO 20250331235959]"
                
                Available search fields in the arXiv API:
                - ti: Title search
                - au: Author search
                - abs: Abstract search
                - cat: Category search
                - submittedDate: Date range in format [YYYYMMDD000000 TO YYYYMMDD235959]
                
                Common arXiv categories:
                - cs.AI: Artificial Intelligence
                - cs.LG: Machine Learning
                - cs.CL: Computation and Language
                - cs.CV: Computer Vision
                - stat: Statistics
                - quant-ph: Quantum Physics
                - physics: Physics
                - math: Mathematics
                - cond-mat: Condensed Matter               
                
                Return ONLY the converted syntax, nothing else.
                """
            
            # Create the chat request
            conversion_request = GroqChatRequest(
                model=model_name,
                messages=[
                    GroqMessage(role="system", content=formatted_prompt),
                    GroqMessage(role="user", content=user_query)
                ],
                temperature=0.1  # Low temperature for more deterministic results
            )
            
            # Get the converted syntax from Groq
            conversion_response = await groq_client(conversion_request)
            
            # Debug the response structure
            logger.debug(f"Response type: {type(conversion_response)}")
            logger.debug(f"Response structure: {conversion_response.model_dump()}")
            
            # Extract the content safely
            try:
                raw_content = conversion_response.choices[0].message.content.strip()
            except (AttributeError, IndexError, KeyError) as extract_error:
                logger.error(f"Error extracting content: {extract_error}")
                logger.error(f"Response structure: {conversion_response}")
                # Try alternative access patterns
                try:
                    raw_content = conversion_response.choices[0]["message"]["content"].strip()
                except (AttributeError, IndexError, KeyError, TypeError):
                    try:
                        # Last resort
                        raw_content = str(conversion_response.choices[0])
                    except (AttributeError, IndexError):
                        raw_content = "quantum computing"  # Default fallback
            
            logger.info(f"Raw LLM output: {raw_content}")
            
            # Extract just the arXiv syntax, not the thinking process
            # Look for the actual query after any <think> tags or at the end of the response
            arxiv_syntax = raw_content
            
            # If there's a <think> tag, extract only what comes after it
            think_pattern = r'</think>\s*(.*?)$'
            think_match = re.search(think_pattern, raw_content, re.DOTALL)
            if think_match:
                arxiv_syntax = think_match.group(1).strip()
            else:
                # If no think tags, try to find the last line that looks like a query
                lines = raw_content.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('<') and not line.startswith('#') and not line.startswith('//'):
                        arxiv_syntax = line
                        break
            
            logger.info(f"Extracted arXiv syntax: {arxiv_syntax}")
            
            # Step 2: Search arXiv with the converted syntax
            arxiv_request = ArXivRequest(
                query=arxiv_syntax,
                max_results=10,
                sort_by="relevance",
                sort_order="descending"
            )
            
            # Get results from arXiv
            arxiv_results = await arxiv_client(arxiv_request)
            
            # Step 3: Save the raw results to a file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = RESULTS_DIR / f"arxiv_results_{timestamp}.json"
            
            # Save as JSON, handling datetime serialization
            with open(result_file, "w") as f:
                # Custom serialization for datetime objects
                json_data = arxiv_results.model_dump()
                json.dump(json_data, f, default=str, indent=2)
            
            logger.info(f"Saved results to {result_file}")
            
            return arxiv_results
            
        except Exception as e:
            logger.error(f"Error in Groq LLM: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty result on error
            return ArXivSearchResult(
                query=user_query,
                total_results=0,
                papers=[]
            )
    
    return query_arxiv 