# PDF Query Tool

The PDF Query Tool enables agents to retrieve relevant content from PDF documents via URL using vector similarity search.

## Overview

This tool provides intelligent PDF document querying by:
1. **Automatic Download**: Fetches PDF documents from URLs on first access
2. **Content Parsing**: Extracts and chunks text content from PDFs
3. **Vector Indexing**: Stores document chunks in a local Qdrant vector database
4. **Similarity Search**: Retrieves the most relevant passages for a given query
5. **Caching**: Reuses indexed documents for subsequent queries to the same URL

## Architecture

### Components

- **PDFClient** (`lits.clients.pdf_client`): Handles PDF downloading, parsing, chunking, and vector storage
- **PDFQueryTool** (`lits.tools.pdf_tools`): Tool interface for agent integration

### Vector Storage

The tool uses Qdrant as a local vector database with:
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **Distance Metric**: Cosine similarity
- **Storage**: Local filesystem (default: `./qdrant_local`)

### Text Processing

- **Chunk Size**: 500 characters (configurable)
- **Chunk Overlap**: 50 characters (configurable)
- **Chunking Strategy**: Attempts to break at sentence boundaries for better context preservation

## Usage

### Basic Setup

```python
from lits.tools import build_tools

# Build PDF tools
tools = build_tools(
    benchmark_name="pdf",
    db_path="./my_pdf_storage"  # Optional: custom storage path
)

pdf_tool = tools[0]
```

### Query a PDF

```python
# Query a PDF document
result = pdf_tool._run(
    url="https://example.com/document.pdf",
    query="What are the main findings?",
    top_k=3  # Return top 3 relevant passages
)

print(result)
```

### Example Output

```
PDF: https://example.com/document.pdf
Query: What are the main findings?
Found 3 relevant passages:

--- Passage 1 (score: 0.856) ---
The main findings indicate that the proposed method achieves 
state-of-the-art performance on benchmark datasets...

--- Passage 2 (score: 0.782) ---
Our key findings demonstrate significant improvements in both 
accuracy and efficiency compared to baseline approaches...

--- Passage 3 (score: 0.745) ---
The experimental results show that the findings are consistent 
across different evaluation metrics...
```

## Configuration

### PDFClient Parameters

```python
from lits.clients.pdf_client import PDFClient

client = PDFClient(
    storage_path="./qdrant_local",           # Vector DB storage path
    collection_name="pdf_documents",          # Qdrant collection name
    embedding_model="all-MiniLM-L6-v2",      # SentenceTransformer model
    chunk_size=500,                           # Max characters per chunk
    chunk_overlap=50                          # Overlap between chunks
)
```

### Tool Parameters

- **url** (required): URL of the PDF document
- **query** (required): Search query string
- **top_k** (optional, default=3): Number of relevant passages to return

## Implementation Details

### First-Time URL Processing

When a URL is encountered for the first time:

1. **Download**: Fetches PDF content via HTTP request
2. **Parse**: Extracts text using PyPDF
3. **Chunk**: Splits text into overlapping segments
4. **Embed**: Generates vector embeddings for each chunk
5. **Index**: Stores chunks with metadata in Qdrant
6. **Cache**: Marks URL as processed

### Subsequent Queries

For previously processed URLs:

1. **Embed Query**: Generates vector embedding for the search query
2. **Search**: Performs cosine similarity search in Qdrant
3. **Filter**: Restricts results to the specified URL
4. **Rank**: Returns top-k most similar chunks

### URL Tracking

- URLs are hashed using MD5 for unique identification
- Processed URLs are cached in memory and persisted in the vector store
- Each chunk is stored with metadata: `url`, `chunk_index`, `text`

## Dependencies

Required packages (automatically installed with lits-llm):
- `pypdf`: PDF parsing
- `qdrant-client`: Vector database
- `sentence-transformers`: Text embeddings
- `requests`: HTTP downloads

## Performance Considerations

### Storage

- Each PDF chunk requires ~384 bytes for embeddings (all-MiniLM-L6-v2)
- A 100-page PDF (~50,000 chars) generates ~100 chunks = ~38KB storage
- Qdrant uses efficient compression and indexing

### Speed

- **First Query**: 2-10 seconds (download + indexing)
- **Subsequent Queries**: <100ms (vector search only)
- **Embedding Generation**: ~10ms per query

### Scalability

- Suitable for 100s-1000s of documents
- For larger scale, consider:
  - Using Qdrant server mode instead of local storage
  - Implementing document expiration policies
  - Using more efficient embedding models

## Error Handling

The tool handles common errors gracefully:

- **Invalid URL**: Raises HTTP error with status code
- **Non-PDF Content**: Validates PDF magic bytes (`%PDF`) and raises clear error message
- **Network Timeout**: 30-second timeout on downloads
- **Storage Issues**: Creates storage directory if missing
- **Invalid Point IDs**: Uses UUID5 for Qdrant-compatible point identifiers

## Advanced Usage

### Custom Embedding Model

```python
from lits.clients.pdf_client import PDFClient

# Use a different embedding model
client = PDFClient(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual support
)
```

### Larger Chunks

```python
# For documents with longer context requirements
client = PDFClient(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Integration with Agents

```python
from lits.agents import create_agent

agent = create_agent(
    tools=build_tools("pdf"),
    llm=your_llm
)

# Agent can now query PDFs autonomously
response = agent.run(
    "Summarize the methodology section from https://example.com/paper.pdf"
)
```

## Limitations

- **Direct PDF URLs Only**: Requires direct links to PDF files (e.g., `https://example.com/paper.pdf`), not web pages or document viewers
- **Text-Only**: Does not extract images, tables, or complex layouts
- **English-Optimized**: Default embedding model works best with English text
- **Local Storage**: Requires disk space for vector database
- **No OCR**: Scanned PDFs without text layer are not supported

## Future Enhancements

Potential improvements:
- Table and figure extraction
- Multi-modal embeddings (text + images)
- Distributed vector storage
- Document update detection
- Metadata extraction (title, authors, date)
