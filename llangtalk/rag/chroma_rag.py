from langchain.vectorstores import Chroma
from .rag_interface import RAG
import logging
import chromadb

logger = logging.getLogger(__name__)


class ChromaRAG(RAG):
    def __init__(self) -> None:
        self.client = chromadb.Client()
