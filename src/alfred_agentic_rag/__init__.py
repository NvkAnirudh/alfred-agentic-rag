"""Alfred Agentic RAG - An agentic RAG system with multiple implementations.

This package provides three different implementations of an agentic RAG system:
- Smolagents: Lightweight agent framework
- LlamaIndex: Data framework for LLM applications
- LangGraph: Graph-based agent orchestration
"""

__version__ = "0.1.0"

# Import implementations
from alfred_agentic_rag.smolagents.agent import SmolagentsRAG
from alfred_agentic_rag.llama_index.agent import LlamaIndexRAG
from alfred_agentic_rag.langgraph.agent import LangGraphRAG
from alfred_agentic_rag.common.config import RAGConfig

__all__ = [
    "SmolagentsRAG",
    "LlamaIndexRAG", 
    "LangGraphRAG",
    "RAGConfig",
]
