import logging
import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List
from llangtalk.rag.rag_interface import RAG

logger = logging.getLogger(__name__)


class FaissRAG(RAG):
    def __init__(self, vectorstore_file: str, metadata_file: str, st_model: str, st_device: str = "cpu"):
        self.vectorstore_file = vectorstore_file
        self.metadata_file = metadata_file
        self.st_model = self.load_st_model(st_model, st_device)
        self.vectorstore = self.load_rag(vectorstore_file, metadata_file)

    def load_rag(self, vectorstore_file: str, metadata_file: str):
        """Load the RAG system from disk, or create a new one if not available."""
        logger.info(f"Loading RAG system from {vectorstore_file} and {metadata_file}")
        try:
            if os.path.exists(vectorstore_file) and os.path.exists(metadata_file):
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)
                vectorstore = FAISS.load_local(
                    vectorstore_file, self.st_model.encode, allow_dangerous_deserialization=True
                )
                vectorstore.docstore._dict.update(metadata)
                logger.info("RAG system loaded successfully.")
                return vectorstore
        except Exception as e:
            logger.warning(f"Error loading existing RAG: {e}")
            logger.info("Creating new RAG system.")

        index = faiss.IndexFlatL2(self.st_model.get_sentence_embedding_dimension())
        docstore = InMemoryDocstore({})
        vectorstore = FAISS(
            embedding_function=self.st_model.encode, index=index, docstore=docstore, index_to_docstore_id={}
        )
        logger.info("New RAG system initialized.")
        return vectorstore

    def add_text_to_rag(self, text: str) -> None:
        """Add text to the RAG system."""
        doc = Document(page_content=text)
        self.vectorstore.add_documents([doc])

    def find_similar(self, query: str, k: int = 5) -> List[Document]:
        """Find similar documents for a query."""
        return self.vectorstore.similarity_search(query, k)

    def similarity_search_by_text(self, query: str, k: int = 5) -> str:
        """Query the RAG system for similar documents."""
        similar_docs = self.find_similar(query, k)
        return "\n".join([doc.page_content for doc in similar_docs])

    def save(self) -> None:
        """Save the RAG system to disk."""
        self.vectorstore.save_local(self.vectorstore_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.vectorstore.docstore._dict, f)
        logger.info("RAG system saved successfully.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize FaissRAG
    rag = FaissRAG(
        vectorstore_file="test_vectorstore",
        metadata_file="test_metadata.pkl",
        st_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Add sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a versatile programming language.",
        "Natural language processing deals with text analysis.",
    ]

    for text in sample_texts:
        rag.add_text_to_rag(text)
        logger.info(f"Added text: {text}")

    # Test similarity search
    query = "Tell me about AI and ML"
    results = rag.similarity_search_by_text(query, k=1)
    print(f"\nQuery: {query}")
    print("Results:", results)

    # Save the index
    rag.save()

    # Reload and verify
    new_rag = FaissRAG(
        vectorstore_file="test_vectorstore",
        metadata_file="test_metadata.pkl",
        st_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    verification_results = new_rag.similarity_search_by_text(query, k=1)
    print("\nVerification Results:", verification_results)
