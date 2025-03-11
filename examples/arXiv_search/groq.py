# examples/arXiv_search/groq.py
import os
import logging
import aiohttp
import ssl
import certifi
from typing import Optional
from dotenv import load_dotenv

from models import GroqChatRequest, GroqChatResponse, GroqLLMFn

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def make_groq_request(request: GroqChatRequest, api_key: str) -> GroqChatResponse:
    """Make an async request to the Groq API"""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=request.model_dump(exclude_none=True),
                ssl=ssl_context
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_message = f"Groq API error (status {response.status}): {error_text}"
                    logger.error(error_message)
                    raise Exception(error_message)
                
                response_json = await response.json()
                return GroqChatResponse(**response_json)
    
    except Exception as e:
        logger.error(f"Failed to communicate with Groq API: {str(e)}")
        raise

def create_groq_client(api_key: Optional[str] = None) -> GroqLLMFn:
    """Factory function to create a Groq client function"""
    # Get API key from parameter or environment
    actual_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not actual_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables or constructor")
    
    async def groq_client(request: GroqChatRequest) -> GroqChatResponse:
        """Send a chat request to the Groq API"""
        return await make_groq_request(request, actual_api_key)
    
    return groq_client