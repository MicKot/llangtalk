from typing import Dict
from click import prompt
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages.ai import AIMessageChunk
from llm_interface import LLMEngine
from langchain.chains.llm import LLMChain
import logging

logger = logging.getLogger(__name__)


class OllamaEngine(LLMEngine):
    def __init__(self, model="llama3.1:8b", chat_version=True, temperature=0.5, seed=42):
        super().__init__()
        if chat_version:
            self.llm_model = LLMChain(
                llm=ChatOllama(model=model, temperature=temperature, seed=seed), memory=self.memory, prompt=self.prompt
            )
            self._chat_version = True
        else:
            self.llm_model = LLMChain(
                OllamaLLM(model=model, temperature=temperature, seed=seed), memory=self.memory, prompt=self.prompt
            )
            self._chat_version = False

    def get_response_content(self, llm_response: AIMessageChunk) -> str:
        return llm_response.content


if __name__ == "__main__":
    ollama = OllamaEngine(model="llama3.1:8b", chat_version=True, temperature=0)

    invoke_response = ollama.invoke("Hello, how are you?")
    ollama.invoke("Can you tell me a joke?")

    ollama.clear_memory()
    invoke_response_second = ollama.invoke("Hello, how are you?")
    assert invoke_response == invoke_response_second
