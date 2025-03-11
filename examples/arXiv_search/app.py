import asyncio
import logging
import sys

from groq_llm import create_groq_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("arxiv_search.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_welcome():
    """Print welcome message and examples"""
    print("\n" + "="*80)
    print("arXiv Search with Natural Language".center(80))
    print("="*80)
    print("\nThis tool converts natural language queries to arXiv search syntax.")
    print("You can ask questions in plain English, and we'll find relevant papers.")
    print("\nExamples:")
    print("  1. Find recent papers on quantum computing")
    print("  2. Papers by John Smith about neural networks")
    print("  3. Latest research on climate change in physics")
    print("\nType 'exit' to quit.")
    print("-"*80)

async def main():
    # Create the LLM
    arxiv_llm = create_groq_llm()
    
    print_welcome()
    
    while True:
        # Get user query
        query = input("\nWhat would you like to search for? ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
        
        try:
            # Process the query
            print("\nSearching arXiv... (this may take a moment)")
            results = await arxiv_llm(query)
            
            # Display results
            print(f"\nFound {results.total_results} papers matching your query:\n")
            
            for i, paper in enumerate(results.papers, 1):
                print(f"{i}. {paper.title}")
                print(f"   Authors: {', '.join(author.name for author in paper.authors)}")
                print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
                print(f"   Categories: {', '.join(paper.categories)}")
                print(f"   PDF: {paper.pdf_url}")
                
                # Truncate summary for readability
                summary = paper.summary.replace("\n", " ")
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                print(f"   Summary: {summary}\n")
            
            if results.total_results == 0:
                print("No papers found. Try refining your search query.")
                
            print("-"*80)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSearch terminated by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nAn unexpected error occurred: {str(e)}") 