import logging
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)


class LLMEngine:

    def __new__(cls, *args, **kwargs):
        # Log when an instance of a subclass is being created
        logging.info(f"Creating an instance of {cls.__name__}")
        # Call the default implementation of __new__
        return super().__new__(cls)

    def __init__(self) -> None:
        self._chat_version = False

    def new_conversation(self):
        self.conversation_chain = ConversationChain(llm=self.llm, verbose=True, memory=ConversationBufferMemory())

    def invoke(self, text: str) -> str:
        pass

    def stream(self, text: str) -> str:
        pass
