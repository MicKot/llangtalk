import logging

logger = logging.getLogger(__name__)


class LLMEngine:

    def __new__(cls, *args, **kwargs):
        # Log when an instance of a subclass is being created
        logging.info(f"Creating an instance of {cls.__name__}")
        # Call the default implementation of __new__
        return super().__new__(cls)

    def __init__(self) -> None:
        pass

    def invoke(self, text: str) -> str:
        pass

    def stream(self, text: str) -> str:
        pass
