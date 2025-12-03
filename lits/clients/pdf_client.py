"""
PDF Client for downloading, parsing, and querying PDF documents via vector similarity.

This client manages PDF documents by:
1. Downloading PDFs from URLs
2. Parsing and chunking content
3. Storing chunks in a local vector database (Qdrant)
4. Performing similarity search for queries
"""

import os
import hashlib
import uuid
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from .base import BaseClient

# Load environment variables
load_dotenv()

try:
    from pypdf import PdfReader
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Missing required dependencies for PDFClient: {e}. "
        "Install with: pip install pypdf qdrant-client sentence-transformers"
    )


class PDFClient(BaseClient):
    """Client for PDF document retrieval and vector-based querying."""

    def __init__(
        self,
        storage_path: str = "./qdrant_local",
        collection_name: str = "pdf_documents",
        embedding_model: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize PDF client with vector storage.

        Args:
            storage_path: Path to store Qdrant database
            collection_name: Name of the Qdrant collection
            embedding_model: SentenceTransformer model name (if None, loads from EMBEDDING_MODEL_NAME env var)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        # Load embedding model from environment if not provided
        if embedding_model is None:
            embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        super().__init__(
            storage_path=storage_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize embedding model
        print("\n" + "="*70)
        print(f"Initializing PDF Client - Loading embedding model: {embedding_model}")
        print("="*70)
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print("="*70)
        print("PDF Client initialization complete!")
        print("="*70 + "\n")
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()
        
        # Track processed URLs
        self.url_cache = set()
        self._load_url_cache()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
            )

    def _load_url_cache(self):
        """Load previously processed URLs from the vector store."""
        try:
            # Scroll through all points to get unique URLs
            offset = None
            while True:
                records, offset = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    if record.payload and "url" in record.payload:
                        self.url_cache.add(record.payload["url"])
                if offset is None:
                    break
        except Exception:
            pass  # Collection might be empty

    def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content = response.content
        
        # Validate that the content is a PDF
        if not content.startswith(b'%PDF'):
            raise ValueError(
                f"URL does not point to a valid PDF file. "
                f"Content type: {response.headers.get('content-type', 'unknown')}. "
                f"Please provide a direct link to a PDF document."
            )
        
        return content

    def _parse_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind(". ")
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
        return [c for c in chunks if c]  # Filter empty chunks

    def _index_document(self, url: str, text: str):
        """Chunk and index document into vector store."""
        chunks = self._chunk_text(text)
        
        # Generate embeddings
        embeddings = self.encoder.encode(chunks, show_progress_bar=False)
        
        # Create points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Use UUID5 for valid UUID point IDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{idx}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "url": url,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
        
        # Upload to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        self.url_cache.add(url)

    def request(self, url: str, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query PDF document with similarity search.

        Args:
            url: URL of the PDF document
            query: Search query
            top_k: Number of top results to return

        Returns:
            Dictionary with relevant chunks and metadata
        """
        # Download and index if new URL
        if url not in self.url_cache:
            pdf_content = self._download_pdf(url)
            text = self._parse_pdf(pdf_content)
            self._index_document(url, text)
        
        # Perform similarity search
        query_embedding = self.encoder.encode(query, show_progress_bar=False)
        
        # Create proper Qdrant filter
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="url",
                    match=MatchValue(value=url)
                )
            ]
        )
        
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
        )
        
        # Format results
        chunks = []
        for result in results:
            chunks.append({
                "text": result.payload["text"],
                "chunk_index": result.payload["chunk_index"],
                "score": result.score,
            })
        
        return {
            "url": url,
            "query": query,
            "chunks": chunks,
            "num_results": len(chunks),
        }

    def ping(self) -> bool:
        """Check if Qdrant client is accessible."""
        try:
            self.qdrant.get_collections()
            return True
        except Exception:
            return False
