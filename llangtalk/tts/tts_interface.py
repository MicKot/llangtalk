import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    def __new__(cls, *args, **kwargs):
        # Log when an instance of a subclass is being created
        logger.info(f"Creating an instance of {cls.__name__}")
        # Call the default implementation of __new__
        return super().__new__(cls)

    def generate_audio_from_text(self, text: str) -> str:
        pass
