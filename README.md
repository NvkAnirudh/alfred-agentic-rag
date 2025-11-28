# Alfred Agentic RAG

An agentic RAG (Retrieval-Augmented Generation) system with three different implementation approaches.

## Overview

This project implements an agentic RAG system using three different frameworks to compare their capabilities and approaches:

1. **Smolagents** - Lightweight agent framework
2. **LlamaIndex** - Comprehensive data framework for LLM applications  
3. **LangGraph** - Graph-based agent orchestration with LangChain

## Project Structure

```
alfred_agentic_rag/
├── src/
│   └── alfred_agentic_rag/
│       ├── smolagents/        # Smolagents implementation
│       │   ├── __init__.py
│       │   └── agent.py
│       ├── llama_index/       # LlamaIndex implementation
│       │   ├── __init__.py
│       │   └── agent.py
│       ├── langgraph/         # LangGraph implementation
│       │   ├── __init__.py
│       │   └── agent.py
│       ├── common/            # Shared utilities
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── utils.py
│       └── __init__.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

## Usage

Each implementation can be used independently:

```python
from alfred_agentic_rag import SmolagentsRAG, LlamaIndexRAG, LangGraphRAG, RAGConfig

# Create configuration
config = RAGConfig(
    model_name="gpt-4",
    embedding_model="text-embedding-3-small",
    top_k=5
)

# Use Smolagents implementation
smolagents_rag = SmolagentsRAG(config)
answer = smolagents_rag.query("What is RAG?")

# Use LlamaIndex implementation
llama_rag = LlamaIndexRAG(config)
answer = llama_rag.query("What is RAG?")

# Use LangGraph implementation
langgraph_rag = LangGraphRAG(config)
answer = langgraph_rag.query("What is RAG?")
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
pytest

# Format code
black src/

# Lint code
ruff check src/

# Type check
mypy src/
```

## Dependencies

- **datasets** - Hugging Face datasets library
- **llama-index** - LlamaIndex framework
- **langchain** - LangChain framework
- **langgraph** - LangGraph for agent orchestration
- **smolagents** - Lightweight agent framework
- **openai** - OpenAI API client
- **tiktoken** - Token counting

## License

MIT
