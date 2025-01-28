import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAG:

    def __init__(self, st_model_path: str, device="cpu"):
        self.st_model = self.load_st_model(st_model_path, device="cpu")

    @staticmethod
    def load_st_model(st_model_path, device="cpu"):
        logger.info(f"Loading SentenceTransformer model from {st_model_path}")
        return SentenceTransformer(st_model_path, device=device)

    def _similarity_search(self, query_vector: np.ndarray, k: int):
        raise NotImplementedError

    def similarity_search_by_text(self, query_text: str, k: int):
        raise NotImplementedError

    def add_text_to_rag(self, text: str, vector: np.ndarray):
        raise NotImplementedError

    def load_or_create_local_rag(self):
        raise NotImplementedError

    def get_embdding_dim(self):
        self.st_model.get_sentence_embedding_dimension()
