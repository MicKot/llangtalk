from operator import index
from langchain.vectorstores import FAISS
from .rag_interface import RAG
import os
import pickle
import logging
import faiss

logger = logging.getLogger(__name__)


class FaissRAG(RAG):
    def __init__(self, vectorstore_file, metadata_file, st_model):
        self.vectorstore_file = vectorstore_file
        self.metadata_file = metadata_file
        self.st_model = self.load_st_model(st_model)
        self.vectorstore = self.load_rag(vectorstore_file, metadata_file)

    def load_rag(
        self,
        vectorstore_file,
        metadata_file,
    ):
        """Load the RAG system from disk, or create a new one if not available."""
        logger.info(f"Loading RAG system from {vectorstore_file} and {metadata_file}")
        if os.path.exists(vectorstore_file) and os.path.exists(metadata_file):
            # Load the vectorstore and associated metadata
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            vectorstore = FAISS.load_local(vectorstore_file, self.st_model.encode)
            vectorstore.docstore._dict.update(metadata)  # Restore metadata
            print("RAG system loaded successfully.")
        else:
            # Initialize a new FAISS vectorstore
            index = faiss.IndexFlatL2(self.st_model.get_sentence_embedding_dimension())

            vectorstore = FAISS(
                embedding_function=self.st_model.encode, index=index, docstore={}, index_to_docstore_id={}
            )
            print("New RAG system initialized.")
        return vectorstore

    def add_text_to_rag(self, text):
        self.vectorstore.add_texts([text])

    def find_similar(self, query, k=5):
        return self.vectorstore.similarity_search(query, k)

    def query(self, query, k=5):
        """Query the RAG system for similar documents."""
        similar_docs = self.find_similar(query, k)
        context = "\n".join([doc.page_content for doc in similar_docs])
        return context

    def save(self, full_save=False):
        """Save the RAG system to disk."""
        if full_save:
            self.vectorstore.save_local(self.vectorstore_file)
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.vectorstore.docstore._dict, f)
            logger.info(f"RAG system saved to {self.vectorstore_file} and {self.metadata_file}")
        else:
            self.vectorstore.save_index(self.vectorstore_file)
            logger.info(f"RAG index saved to {self.vectorstore_file}")
