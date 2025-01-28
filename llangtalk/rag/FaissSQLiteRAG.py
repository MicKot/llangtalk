import sqlite3
import faiss
import numpy as np
import os
from llangtalk.rag.rag_interface import RAG
import logging

logger = logging.getLogger(__name__)


class FaissSQLiteRAG(RAG):

    def __init__(
        self,
        db_path: str,
        index_path: str = "faiss.index",
        st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        st_device: str = "cpu",
    ):
        self.st_model = self.load_st_model(st_model_path=st_model, device=st_device)
        self.db_path = db_path
        self.vector_dim = self.get_embdding_dim()
        self.index_path = index_path

    def load_or_create_local_rag(self):
        self.index = self._load_or_create_index()
        self._create_database()

    def _load_or_create_index(self):

        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        else:
            return faiss.IndexFlatL2(self.vector_dim)

    def _create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                vector BLOB NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def add_text_to_rag(self, text: str, vector: np.ndarray):
        # Ensure vector is float32
        vector = vector.astype(np.float32)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents (text, vector) VALUES (?, ?)", (text, vector.tobytes()))
        conn.commit()
        conn.close()
        self.index.add(vector.reshape(1, -1))
        self._save_index()

    def _similarity_search(self, query_vector: np.ndarray, k: int = 5):
        # Ensure query vector is float32
        query_vector = query_vector.astype(np.float32)
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []

        for idx in indices[0]:
            if idx == -1:
                continue
            cursor.execute(
                "SELECT text FROM documents WHERE id = ?",
                (
                    str(
                        idx + 1,
                    ),
                ),
            )
            result = cursor.fetchone()
            if result:
                # Get text from result tuple
                results.append(result[0])

        conn.close()
        return results

    def _check_database_contents(self):
        "DEBUG method to check database contents"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text FROM documents")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def embed_text(self, text: str) -> np.ndarray:
        embeddings = self.st_model.encode(text)
        return embeddings

    def similarity_search_by_text(self, text: str, k: int = 5):
        query_vector = self.embed_text(text)
        return self._similarity_search(query_vector, k)


# Example usage
if __name__ == "__main__":
    rag = FaissSQLiteRAG(db_path="rag.db", vector_dim=384)  # Adjust vector_dim based on the model used

    # Example texts
    text1 = "This is a sample document."
    text2 = "This is another sample document."
    query_text = "This is a document."

    # Add texts with vectors
    vector1 = rag.embed_text(text1)
    vector2 = rag.embed_text(text2)
    rag.add_text_to_rag(text1, vector1)
    rag.add_text_to_rag(text2, vector2)

    # Perform similarity search by text
    results = rag.similarity_search_by_text(query_text, 1)
    print("Similarity Search Results:", results)

    # Check database contents
    db_contents = rag.check_database_contents()
    print("\nDatabase Contents:")
    for row in db_contents:
        print(f"ID: {row[0]}, Text: {row[1]}")
