import logging
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)


class LLMEngine:
    def __new__(cls, *args, **kwargs):
        logging.info(f"Creating an instance of {cls.__name__}")
        return super().__new__(cls)

    def __init__(self) -> None:
        self._chat_version = False
        self.llm_model: BaseChatModel = None
        # Initialize memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        # Create prompt template with memory
        self.prompt = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

    def invoke(self, text: str) -> str:
        response = self.llm_model.invoke(text)
        return response["text"]

    def stream(self, text: str):
        response = self.llm_model.stream(text)

        # Collect full response for memory
        full_response = ""
        for chunk in response:
            print(chunk["text"])
            full_response += chunk["text"]
            yield chunk["text"]

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_history(self) -> List[dict]:
        return self.memory.load_memory_variables({})["chat_history"]
