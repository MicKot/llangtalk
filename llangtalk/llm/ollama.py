from langchain_ollama import OllamaLLM
from .llm_interface import LLMEngine
import logging

logger = logging.getLogger(__name__)


class Ollama(LLMEngine):
    def __init__(self, model="llama3.1:8b"):
        self.llm = OllamaLLM(model=model)

    def invoke(self, text: str) -> str:
        return self.llm.invoke(text)

    def stream(self, text: str) -> str:
        return self.llm.stream(text)
