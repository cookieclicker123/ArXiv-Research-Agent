# arXiv Search Example

This example demonstrates how to convert natural language queries into structured arXiv API queries, retrieve relevant papers, and present the results.

## Overview

This proof-of-concept implements:

1. Natural language query processing using Groq LLM
2. Conversion of natural language to arXiv search syntax
3. Searching arXiv using their API
4. Returning formatted results with paper links

## How it Works

1. User enters a natural language query (e.g., "Find me recent papers on quantum computing applications in finance")
2. Groq LLM converts this to arXiv search syntax (e.g., "cat:quant-ph AND (quantum computing) AND (finance OR financial) AND submittedDate:[20230101 TO 20250331]")
3. arXiv API is queried with the converted syntax
4. Results are returned and saved to the `arXiv_results` folder
5. LLM formats a user-friendly response with paper summaries and links

## arXiv Search Syntax

arXiv uses a specific search syntax documented at [arXiv API User Manual](https://arxiv.org/help/api/user-manual).

Key search operators:
- Boolean operators: AND, OR, NOT
- Field-specific search: ti (title), au (author), abs (abstract), cat (category)
- Date range: submittedDate:[YYYYMMDD TO YYYYMMDD]
- Exact phrase matching with quotes: "quantum computing"

Example conversions:
- "Recent papers on quantum computing" → "cat:quant-ph AND (quantum computing) AND submittedDate:[20230101 TO 20250331]"
- "Papers by John Smith about neural networks" → "au:Smith_J AND (neural networks)"

## Setup

1. Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

some examples:

Papers by Geoffrey Hinton about deep learning
Recent research on large language models in both computer science and statistics
Find papers with "transformer" in the title related to natural language processing
Papers about reinforcement learning that don't mention robotics
Papers on climate modeling published in the first half of 2024
Research combining quantum physics and machine learning
ICML 2023 papers about diffusion models


```bash
cd examples/arXiv_search
python app.py
```

## Output

Results are saved to the `arXiv_results` folder with:
- Raw JSON responses from arXiv API
- Formatted summaries from the LLM

## Limitations

- arXiv API has rate limits (no more than 1 request per 3 seconds)
- Search capabilities are limited compared to semantic search
- Results quality depends on the LLM's ability to convert natural language to arXiv syntax
