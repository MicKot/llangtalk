import logging
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    def invoke(self, text: str) -> str:
        # Get chat history from memory
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # Create chain with memory
        response = self.llm_model.invoke(self.prompt.format_messages(chat_history=chat_history, input=text))

        # Save to memory
        self.memory.save_context({"input": text}, {"output": response.content})

        return response.content

    def stream(self, text: str):
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        response = self.llm_model.stream(self.prompt.format_messages(chat_history=chat_history, input=text))

        # Collect full response for memory
        full_response = ""
        for chunk in response:
            full_response += chunk.content
            yield chunk.content

        self.memory.save_context({"input": text}, {"output": full_response})

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_history(self) -> List[dict]:
        return self.memory.load_memory_variables({})["chat_history"]
