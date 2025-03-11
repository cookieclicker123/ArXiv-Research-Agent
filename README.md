# Research Assistant

A tool to enhance research productivity by providing intelligent paper discovery, organization, and analysis capabilities.

## Vision

Research Assistant aims to transform how researchers, students, and curious minds interact with academic literature. By combining powerful search capabilities with intelligent analysis tools, it helps users:

- Discover relevant papers more efficiently through semantic search
- Organize research materials in a structured, searchable way
- Extract key insights from papers without reading every word
- Identify connections between papers and research trends
- Generate literature reviews and research summaries automatically

The system integrates with arXiv to access a vast repository of academic papers, processes them for both textual and semantic search, and provides tools to analyze and synthesize information across multiple papers.

## Incremental Development Plan

### Phase 1: Core arXiv Integration (Week 1-2)

1. **Create minimal arXiv client**
   - Implement paper search using arXiv API (via arxiv Python package)
   - Fetch metadata (ID, title, authors, abstract, categories, published date)
   - Download PDF and extract full text using PyPDF2 or pdfplumber
   - Store complete paper data in JSON format
   - Handle rate limiting with exponential backoff
   - Create command-line demo that saves search results locally

2. **Design simple paper model**
   - Create Paper class using Pydantic BaseModel:
   ```python
   from pydantic import BaseModel, Field
   from typing import List, Optional, Dict, Any
   from datetime import datetime

   AuthorList = List[str]
   CategoryList = List[str]

   class Paper(BaseModel):
       id: str
       title: str
       authors: AuthorList
       abstract: str
       categories: CategoryList
       published_date: datetime
       pdf_url: str
       full_text: Optional[str] = None
       embeddings: Optional[Dict[str, Any]] = None
       
       class Config:
           json_schema_extra = {
               "example": {
                   "id": "2307.09288",
                   "title": "Quantum Computing: Recent Advances and Future Directions",
                   "authors": ["Smith, J.", "Johnson, A."],
                   "abstract": "This paper reviews recent developments in quantum computing...",
                   "categories": ["quant-ph", "cs.AI"],
                   "published_date": "2023-07-18T14:22:10Z",
                   "pdf_url": "https://arxiv.org/pdf/2307.09288.pdf"
               }
           }
   ```
   - Implement JSON serialization/deserialization methods
   - Add utility functions for text cleaning and normalization

3. **Implement local storage**
   - Create file-based storage system using directory structure:
   ```bash
   research_assistant/
   ├── data/
   │   ├── papers/
   │   │   ├── <paper_id>/
   │   │   │   ├── metadata.json  # Basic paper metadata with timestamps
   │   │   │   ├── fulltext.json  # Complete extracted text
   │   │   │   └── <paper_id>.faiss  # FAISS index for this specific paper
   │   ├── collections/
   │   │   ├── <collection_id>/
   │   │   │   └── paper_ids.json  # List of paper IDs in collection
   │   └── search_index/
   │       ├── global.faiss  # Global FAISS index for all papers
   │       └── id_mapping.json  # Maps vector IDs to paper IDs
   ```
   - Implement CRUD operations for papers and collections
   - Ensure consistent object IDs link metadata, full text, and vector embeddings

### Phase 2: Search and Retrieval (Week 3-4)

1. **Add text embeddings**
   - Integrate sentence-transformers for generating embeddings
   - Create embeddings for title, abstract, and full text separately
   - Store embeddings in numpy arrays alongside JSON data
   - Implement batch processing for generating embeddings

2. **Implement semantic search**
   - Create vector similarity search using cosine similarity
   - Build hybrid search combining keyword and semantic matching
   - Implement relevance scoring algorithm
   - Create search API with filtering options:
   ```python
   def search_papers(query, filters=None, limit=10, semantic_weight=0.7):
       """
       Search papers using hybrid semantic and keyword matching
       
       Args:
           query (str): The search query
           filters (dict): Optional filters like author, category, date_range
           limit (int): Maximum number of results to return
           semantic_weight (float): Weight for semantic vs keyword search (0-1)
           
       Returns:
           List of Paper objects sorted by relevance
       """
   ```

3. **Create paper collection management**
   - Implement collections as user-defined groups of papers
   - Add tagging system with custom and automatic tags
   - Create import from BibTeX/CSV functionality
   - Build export to BibTeX/CSV/JSON functionality

### Phase 3: Analysis and Synthesis (Week 5-6)

1. **Add paper summarization**
   - Integrate with local LLM via Ollama for summarization
   - Implement prompt templates for different summary types:
     - Brief summary (1-2 paragraphs)
     - Detailed summary (4-5 paragraphs)
     - Technical summary (focusing on methodology)
     - Simplified summary (for non-experts)
   - Extract key points, contributions, and limitations

2. **Implement paper comparison**
   - Create side-by-side comparison of 2+ papers
   - Generate similarity analysis based on embeddings
   - Identify common themes, methods, and conclusions
   - Highlight unique contributions of each paper

3. **Create citation analysis**
   - Extract citations from paper full text using regex patterns
   - Build citation graph using networkx
   - Calculate influence metrics (citation count, h-index)
   - Visualize citation relationships with matplotlib

### Phase 4: User Experience (Week 7-8)

1. **Build simple web interface**
   - Create Flask/FastAPI web application
   - Implement paper search, viewing, and organization UI
   - Add collection management interface
   - Create simple dashboard with recent papers and statistics

2. **Add user authentication**
   - Implement JWT-based authentication
   - Create user registration and login flows
   - Store user preferences and search history
   - Add access control for private collections

3. **Create research workspace**
   - Implement Markdown-based note-taking linked to papers
   - Create reading lists with progress tracking
   - Add annotation capabilities for papers
   - Build export functionality to Markdown/PDF/HTML

### Phase 5: Advanced Features (Week 9-10)

1. **Implement author tracking**
   - Create author profiles with publication history
   - Set up scheduled checks for new papers by tracked authors
   - Generate email notifications for new publications
   - Analyze co-authorship networks and visualize collaborations

2. **Add research trend identification**
   - Analyze keyword frequency over time periods
   - Track growth/decline of research categories
   - Identify emerging topics using embedding clusters
   - Generate trend reports with visualizations

3. **Create literature review generation**
   - Implement structured literature review templates
   - Generate comparative analysis across multiple papers
   - Create citation network visualization
   - Add export to LaTeX for academic papers

### Phase 6: Refinement and Scaling (Week 11-12)

1. **Optimize performance**
   - Implement caching layer for frequent searches
   - Optimize embedding generation and storage
   - Add background processing for long-running tasks
   - Improve search algorithm efficiency

2. **Add advanced customization**
   - Create custom search filters and saved searches
   - Implement personalized paper recommendations
   - Add customizable notification settings
   - Create workflow templates for different research styles

3. **Prepare for scaling**
   - Migrate from file-based storage to MongoDB for metadata and full text
   - Implement Qdrant for vector search capabilities
   - Add proper error handling and logging
   - Create Docker configuration for easy deployment

## Database Architecture

As the project scales, we'll transition from file-based storage to a dual-database architecture:

### MongoDB
- Store paper metadata and full text
- Handle structured queries (author, title, date, etc.)
- Manage user data, collections, and annotations
- Support full-text search capabilities

### Qdrant
- Store vector embeddings for semantic search
- Handle similarity queries and nearest neighbor search
- Support hybrid search combining semantic and keyword matching
- Enable clustering and topic modeling

This separation allows us to leverage the strengths of each database type while maintaining a unified API for the application.

## Development Principles

- **Working code first**: Implement real functionality before abstractions
- **Test with real data**: Use actual arXiv papers throughout development
- **Incremental complexity**: Add features only when previous ones work well
- **User feedback driven**: Test with real research workflows frequently
- **Modular design**: Keep components loosely coupled for flexibility

## Getting Started

```bash
git clone https://github.com/username/research-assistant.git
cd research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install --upgrade pip

# create .env file
touch .env

# Install dependencies
pip install -r requirements.txt
```

## Paper Discovery and Retrieval Strategy

### Challenge: Bridging Natural Language Queries and API Limitations

Most academic paper APIs (including arXiv) don't natively support semantic or natural language search. To address this gap, we'll implement a multi-stage search strategy:

#### For Existing Papers in Our Database:
1. Use vector similarity search via FAISS/Qdrant to find semantically relevant papers
2. Apply metadata filters (date, author, etc.) to refine results
3. Rank results by combined semantic and metadata relevance

#### For New Paper Discovery:
1. **Query Translation Layer**:
   - Parse natural language queries using NLP techniques
   - Extract key search terms, filters, and intent
   - Translate to structured queries compatible with arXiv API
   - Example: "recent papers on quantum computing applications" →
     ```python
     {
         "search_query": "cat:quant-ph AND (quantum computing) AND (application OR applications)",
         "sort_by": "submittedDate",
         "sort_order": "descending",
         "max_results": 50
     }
     ```

2. **Hybrid Search Approach**:
   - Send translated query to arXiv API
   - Download and process results (metadata + PDF URL --> full text extraction)
   - Generate embeddings for new papers
   - Perform secondary semantic ranking on results
   - Present to user with relevance explanation

3. **Continuous Learning**:
   - Track which results users find valuable
   - Improve query translation based on user feedback
   - Build a library of effective query patterns

### API Integration Options

1. **Primary: arXiv API**
   - Pros: Comprehensive repository, reliable, free
   - Cons: Limited search capabilities, no semantic search
   - Implementation: Use arxiv Python package with our query translation layer

2. **Secondary: Semantic Scholar API**
   - Pros: More metadata, citation information, some influence metrics
   - Cons: Similar search limitations, rate limits
   - Implementation: Supplement arXiv data with citation information

3. **Fallback: Controlled Web Scraping**
   - For cases where API limitations prevent finding relevant papers
   - Target specific academic search engines with paper-focused results
   - Implement strict filtering to only retrieve arXiv/academic papers
   - Use sparingly and respect robots.txt and rate limits

This multi-layered approach allows us to provide a natural language interface to users while working within the constraints of available APIs.

## Database Architecture Evolution

### Phase 1: File-based Storage with ID Linking
- Each paper has a unique ID (derived from arXiv ID when available)
- All files for a paper (metadata, full text, embeddings) share this ID
- Global FAISS index maps vector IDs back to paper IDs
- Example:
  ```
  papers/2307.09288/metadata.json
  papers/2307.09288/fulltext.json
  papers/2307.09288/2307.09288.faiss
  ```

### Phase 2: Transition to MongoDB + Qdrant
- MongoDB stores metadata and full text with the same ID
- Qdrant stores vector embeddings with the same ID as reference
- Queries typically enter through Qdrant for semantic matching
- Results are enriched with metadata from MongoDB
- Both databases remain synchronized via the shared ID system

This architecture ensures that semantic search capabilities work for both existing papers in our database and newly discovered papers through the API translation layer.

# Run the Arxiv Search Example

```bash
python examples/arXiv_search/groq_llm.py
```

this will show you the api in action, converting your natural language reuests to arXiv native queries, giving us access to the full arXiv paper database. the fact the LLM is used to convert the natural language query to an arXiv native query is a key part of the system, so its imortant to exemplify this in a single example.



## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


