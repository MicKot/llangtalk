import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAG:
    @staticmethod
    def load_st_model(st_model_path):
        logger.info(f"Loading SentenceTransformer model from {st_model_path}")
        return SentenceTransformer(st_model_path)
