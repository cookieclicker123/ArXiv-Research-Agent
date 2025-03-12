# ArXiv Explorer

A semantic search and personal library system for arXiv papers.

## Overview

ArXiv Explorer allows users to:

1. Search arXiv papers using natural language queries
2. Build a personal library of research papers
3. Perform semantic search across papers
4. Get AI-powered summaries and insights

## Current Status

Phase 1 (arXiv Search) is complete and working in the examples directory.

## Development Roadmap

### Phase 1: arXiv Search ✅
- Convert natural language to arXiv search syntax
- Query arXiv API with converted syntax
- Return formatted results with paper metadata
- Save results for further processing

### Phase 2: PDF Processing & Embedding
- Download PDFs from arXiv links
- Extract text with PyMuPDF
- Generate embeddings for semantic search
- Create FAISS indices with paper ID tagging
- Implement chunking strategies for better search

### Phase 3: Local Knowledge Base
- Build metadata search using JSON files
- Implement vector search using FAISS indices
- Create combined search ranking
- Add filtering capabilities (date, author, etc.)
- Develop query expansion for better matches

### Phase 4: Database Integration
- Integrate MongoDB for metadata storage
- Implement Qdrant for vector storage
- Ensure consistent ID scheme across systems
- Add caching layer for frequent queries
- Design proper indexing strategy

### Phase 5: User Management & Integration
- Implement authentication and authorization
- Create personal paper libraries
- Develop agent orchestration system
- Build result synthesis and formatting
- Add paper recommendations

### Phase 6: Production Readiness
- FastAPI implementation
- Comprehensive testing
- Docker containerization
- CI/CD pipeline
- Monitoring and logging
- Performance optimization

## Technical Architecture

The system uses:
- Python 3.10+
- FastAPI for the backend
- MongoDB for metadata storage
- Qdrant for vector embeddings
- PyMuPDF for PDF processing
- HTTPX for async HTTP requests
- Groq LLM API for natural language processing

## Getting Started

### Prerequisites
- Python 3.10+
- API keys for Groq

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arxiv-explorer.git
cd arxiv-explorer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
```

## Running the Examples

### arXiv Search

```bash
cd examples/arXiv_search
python app.py
```

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

