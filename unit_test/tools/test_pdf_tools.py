"""
Unit tests for PDF Query Tool.

Simple, production-like tests using real PDF downloads.
Tests the complete workflow: download -> parse -> chunk -> index -> query


python -m pytest unit_test/test_pdf_tools.py::TestPDFClientBasics -v
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lits.clients.pdf_client import PDFClient
from lits.tools.pdf_tools import PDFQueryTool


class TestPDFClientBasics(unittest.TestCase):
    """Test basic PDFClient functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_client_initialization(self):
        """Test PDFClient initializes correctly."""
        client = PDFClient(storage_path=self.temp_dir)
        
        self.assertEqual(client.storage_path, Path(self.temp_dir))
        self.assertEqual(client.collection_name, "pdf_documents")
        self.assertEqual(client.chunk_size, 500)
        self.assertEqual(client.chunk_overlap, 50)
        self.assertIsNotNone(client.encoder)
        self.assertIsNotNone(client.qdrant)

    def test_url_to_id(self):
        """Test URL hashing for unique IDs."""
        client = PDFClient(storage_path=self.temp_dir)
        
        url1 = "https://example.com/doc1.pdf"
        url2 = "https://example.com/doc2.pdf"
        
        id1 = client._url_to_id(url1)
        id2 = client._url_to_id(url2)
        
        # IDs should be different for different URLs
        self.assertNotEqual(id1, id2)
        
        # Same URL should produce same ID
        self.assertEqual(id1, client._url_to_id(url1))

    def test_chunk_text(self):
        """Test text chunking with overlap."""
        client = PDFClient(
            storage_path=self.temp_dir,
            chunk_size=50,
            chunk_overlap=10
        )
        
        text = "This is a test sentence. " * 10  # ~250 chars
        chunks = client._chunk_text(text)
        
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be non-empty and within size limit
        for chunk in chunks:
            self.assertTrue(chunk)
            self.assertLessEqual(len(chunk), 60)  # Allow some flexibility

    def test_ping(self):
        """Test client connectivity check."""
        client = PDFClient(storage_path=self.temp_dir)
        self.assertTrue(client.ping())


class TestPDFRealDownload(unittest.TestCase):
    """Test with real PDF download from arXiv."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = "temp/"
        self.test_url = "https://arxiv.org/pdf/2509.25835.pdf"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_download_real_pdf(self):
        """Test downloading a real PDF from arXiv."""
        client = PDFClient(storage_path=self.temp_dir)
        
        # Download PDF
        pdf_content = client._download_pdf(self.test_url)
        
        # Verify it's a PDF (starts with PDF magic bytes)
        self.assertTrue(pdf_content.startswith(b'%PDF'))
        self.assertGreater(len(pdf_content), 1000)  # Should be substantial

    def test_parse_real_pdf(self):
        """Test parsing a real PDF."""
        client = PDFClient(storage_path=self.temp_dir)
        
        # Download and parse
        pdf_content = client._download_pdf(self.test_url)
        text = client._parse_pdf(pdf_content)
        
        # Verify text was extracted
        self.assertGreater(len(text), 100)
        self.assertIsInstance(text, str)

    def test_full_workflow_first_query(self):
        """Test complete workflow: download, parse, index, query."""
        client = PDFClient(storage_path=self.temp_dir)
        
        # First query - should download and index
        result = client.request(
            url=self.test_url,
            query="guide me to extend CiT to other search algorithms",
            top_k=2
        )
        
        # Verify result structure
        self.assertEqual(result['url'], self.test_url)
        self.assertEqual(result['query'], "guide me to extend CiT to other search algorithms")
        self.assertGreater(result['num_results'], 0)
        self.assertLessEqual(result['num_results'], 2)
        
        # Verify chunks have expected fields
        for chunk in result['chunks']:
            self.assertIn('text', chunk)
            self.assertIn('chunk_index', chunk)
            self.assertIn('score', chunk)
            self.assertGreater(len(chunk['text']), 0)
            self.assertGreaterEqual(chunk['score'], 0)
            self.assertLessEqual(chunk['score'], 1)

    def test_cached_query(self):
        """Test that second query uses cached document."""
        client = PDFClient(storage_path=self.temp_dir)
        
        # First query
        result1 = client.request(
            url=self.test_url,
            query="machine learning",
            top_k=1
        )
        
        # Verify URL is now cached
        self.assertIn(self.test_url, client.url_cache)
        
        # Second query - should use cache
        result2 = client.request(
            url=self.test_url,
            query="neural networks",
            top_k=1
        )
        
        # Both should return results
        self.assertGreater(result1['num_results'], 0)
        self.assertGreater(result2['num_results'], 0)


class TestPDFQueryTool(unittest.TestCase):
    """Test PDFQueryTool with real PDF."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://arxiv.org/pdf/2509.25835.pdf"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_initialization(self):
        """Test PDFQueryTool initializes correctly."""
        client = PDFClient(storage_path=self.temp_dir)
        tool = PDFQueryTool(client=client)
        
        self.assertEqual(tool.name, "query_pdf")
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.args_schema)

    def test_tool_with_real_pdf(self):
        """Test tool execution with real PDF."""
        client = PDFClient(storage_path=self.temp_dir)
        tool = PDFQueryTool(client=client)
        
        # Execute query
        result = tool._run(
            url=self.test_url,
            query="What is this paper about?",
            top_k=2
        )
        
        # Verify output format
        self.assertIn(self.test_url, result)
        self.assertIn("What is this paper about?", result)
        self.assertIn("Passage 1", result)
        self.assertIn("score:", result)
        
        # Should contain actual content
        self.assertGreater(len(result), 200)

    def test_tool_args_schema(self):
        """Test tool argument schema validation."""
        from lits.tools.pdf_tools import PDFQueryInput
        
        # Valid input
        valid_input = PDFQueryInput(
            url='https://example.com/test.pdf',
            query='test query',
            top_k=5
        )
        self.assertEqual(valid_input.url, 'https://example.com/test.pdf')
        self.assertEqual(valid_input.query, 'test query')
        self.assertEqual(valid_input.top_k, 5)
        
        # Default top_k
        default_input = PDFQueryInput(
            url='https://example.com/test.pdf',
            query='test query'
        )
        self.assertEqual(default_input.top_k, 3)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration test."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://arxiv.org/pdf/2509.25835.pdf"
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_queries_same_document(self):
        """Test multiple queries to the same document."""
        client = PDFClient(storage_path=self.temp_dir)
        tool = PDFQueryTool(client=client)
        
        # Query 1
        result1 = tool._run(
            url=self.test_url,
            query="methodology",
            top_k=2
        )
        
        # Query 2 - should be faster (cached)
        result2 = tool._run(
            url=self.test_url,
            query="results",
            top_k=2
        )
        
        # Both should return valid results
        self.assertIn("methodology", result1)
        self.assertIn("results", result2)
        self.assertGreater(len(result1), 100)
        self.assertGreater(len(result2), 100)


if __name__ == '__main__':
    unittest.main()
