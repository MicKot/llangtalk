from typing import Dict
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages.ai import AIMessageChunk
from .llm_interface import LLMEngine
import logging

logger = logging.getLogger(__name__)


class Ollama(LLMEngine):
    def __init__(self, model="llama3.1:8b", chat_version=True):
        if chat_version:
            self.llm = ChatOllama(model=model)
            self._chat_version = True
        else:
            self.llm = OllamaLLM(model=model)
            self._chat_version = False

    def invoke(self, text: str) -> str:
        return self.llm.invoke(text)

    def stream(self, text: str) -> str:
        return self.llm.stream(text)

    def get_response_content(self, llm_response: AIMessageChunk) -> str:
        return llm_response.content
