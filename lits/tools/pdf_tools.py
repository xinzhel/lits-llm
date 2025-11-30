"""
PDF Query Tool for retrieving relevant content from PDF documents via URL.

This tool enables agents to:
1. Query PDF documents by URL
2. Automatically download and index new PDFs
3. Retrieve relevant content using vector similarity search
"""

from typing import Type
from pydantic import BaseModel, Field
from .base import BaseTool


class PDFQueryInput(BaseModel):
    """Input schema for PDF query tool."""
    url: str = Field(
        ...,
        description="URL of the PDF document to query"
    )
    query: str = Field(
        ...,
        description="Search query to find relevant content in the PDF"
    )
    top_k: int = Field(
        default=3,
        description="Number of relevant chunks to return (default: 3)"
    )


class PDFQueryTool(BaseTool):
    """
    Query PDF documents from URLs and retrieve relevant content.
    
    This tool automatically:
    - Downloads PDFs from provided URLs (first time only)
    - Parses and indexes the content into a vector database
    - Performs similarity search to find relevant passages
    - Returns the most relevant text chunks for the query
    
    Subsequent queries to the same URL will use the cached indexed version.
    """
    
    name: str = "query_pdf"
    description: str = (
        "Query a PDF document from a URL and retrieve relevant content. "
        "Input: PDF URL and a search query. "
        "Output: Relevant text passages from the PDF that match the query. "
        "The PDF is automatically downloaded and indexed on first use."
    )
    args_schema: Type[BaseModel] = PDFQueryInput

    def _run(self, url: str, query: str, top_k: int = 3, **kwargs) -> str:
        """
        Execute PDF query.

        Args:
            url: URL of the PDF document
            query: Search query
            top_k: Number of results to return

        Returns:
            Formatted string with relevant content
        """
        result = self.client.request(url=url, query=query, top_k=top_k)
        
        # Format output
        output_lines = [
            # f"PDF: {result['url']}",
            # f"Query: {result['query']}",
            # f"Found {result['num_results']} relevant passages:\n",
        ]
        
        for i, chunk in enumerate(result['chunks'], 1):
            output_lines.append(f"--- Retrieved Passage {i} (score: {chunk['score']:.3f}) ---")
            output_lines.append(chunk['text'])
            output_lines.append("")
        
        return "\n".join(output_lines)
