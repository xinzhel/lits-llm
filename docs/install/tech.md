# Technology Stack

## Build System

- **Package Manager**: pip with setuptools
- **Build Backend**: setuptools (PEP 517/518 compliant)
- **Configuration**: `pyproject.toml` (modern Python packaging)
- **Environment Management**: conda (via `environment.yml`)

## Core Dependencies

### LLM Providers
- **OpenAI**: openai>=1.35.10
- **AWS Bedrock**: boto3 (AWS SDK)
- **LiteLLM**: Multi-provider abstraction
- **LangChain**: langchain, langchain-huggingface, langchain-community

### Memory & Vector Stores
- **mem0ai**: Cross-trajectory memory backend
- **Qdrant**: qdrant-client (local vector database)
- **Sentence Transformers**: Embeddings

### Geospatial & Database
- **psycopg2-binary**: PostgreSQL adapter
- **GeoAlchemy2**: Spatial database ORM
- **GeoPandas**: Geospatial data structures

### Document Processing
- **pypdf**: PDF parsing and extraction

## Python Version

- **Required**: Python >= 3.11
