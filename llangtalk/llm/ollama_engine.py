from typing import Dict
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages.ai import AIMessageChunk
from llm_interface import LLMEngine
import logging

logger = logging.getLogger(__name__)


class OllamaEngine(LLMEngine):
    def __init__(self, model="llama3.1:8b", chat_version=True):
        super().__init__()
        if chat_version:
            self.llm_model = ChatOllama(model=model)
            self._chat_version = True
        else:
            self.llm_model = OllamaLLM(model=model)
            self._chat_version = False

    def get_response_content(self, llm_response: AIMessageChunk) -> str:
        return llm_response.content


if __name__ == "__main__":
    import time

    # Initialize the Ollama instance
    ollama = OllamaEngine(model="llama3.1:8b", chat_version=True)

    # Function to stream responses
    def stream_responses(input_text):
        print(f"User: {input_text}")
        for chunk in ollama.stream(input_text):
            print(f"{chunk}", end="", flush=True)
            time.sleep(0.1)  # Simulate streaming delay
        print()  # Newline after the response

    # Stream a couple of times
    stream_responses("Hello, how are you?")
    stream_responses("Can you tell me a joke?")

    # Check memory
    conversation_history = ollama.get_conversation_history()
    print("\nConversation History:")
    for message in conversation_history:
        print(f"{message.content}")

    # Clear memory
    ollama.clear_memory()
    print("\nMemory cleared.")
