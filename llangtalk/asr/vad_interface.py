from random import sample
from venv import logger
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VADEngine:
    def __init__(
        self, sample_rate=16000, speech_threshold=0.5, post_speech_timeout=0.5, chunk_size=0.02, min_speech_time=0.1
    ):
        logger.info("Loading VAD model")
        self.model, self.utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
        self.sample_rate = sample_rate
        self.internal_audio_buffer = torch.zeros(0, dtype=torch.float32)
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        self.speaking = False
        self.speech_threshold = speech_threshold
        self.post_speech_timeout = post_speech_timeout
        self.POST_SPEECH_TIMEOUT = False
        self.first_chunk_with_speech = None
        self.current_chunk_idx = 0
        self.chunk_size = chunk_size
        self.min_speech_time = min_speech_time

    def process_chunk(self, audio_chunk: np.ndarray):
        self.current_chunk_idx += 1
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk)
        audio_chunk = audio_chunk.squeeze()
        self.internal_audio_buffer = torch.concatenate([self.internal_audio_buffer, audio_chunk])
        for i in range(0, len(self.internal_audio_buffer), self.window_size_samples):
            chunk = self.internal_audio_buffer[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                self.internal_audio_buffer = audio_chunk[i:]
                break
            speech_prob = self.model(chunk, self.sample_rate).item()
            logger.debug(f"Speech probability: {speech_prob}")
            is_speech = speech_prob > self.speech_threshold
            current_time = self.current_chunk_idx * self.chunk_size

            if is_speech:
                self.last_speech_time = current_time
                self.speaking = True
                self.first_chunk_with_speech = self.first_chunk_with_speech or self.current_chunk_idx
                logger.debug(f"Speech detected at {current_time}")
            elif (
                self.speaking
                and self.last_speech_time
                and current_time - self.last_speech_time > self.post_speech_timeout
            ):
                if current_time - self.last_speech_time > self.min_speech_time:
                    logger.debug(f"Post speech timeout at {current_time}")
                    self.POST_SPEECH_TIMEOUT = True
                else:
                    logger.debug(f"Speech too short, ignoring")
                self.speaking = False

    def reset_states(self):
        logger.debug("Resetting VAD states")
        self.POST_SPEECH_TIMEOUT = False
        self.model.reset_states()
        self.audio_len = 0
        self.internal_audio_buffer = torch.zeros(0, dtype=torch.float32)
        self.speaking = False
        self.last_speech_time = None
        self.first_chunk_with_speech = None
        self.current_chunk_idx = 0

    def soft_reset(self):
        logger.debug("Soft resetting VAD")
        self.POST_SPEECH_TIMEOUT = False
        self.internal_audio_buffer = self.internal_audio_buffer[-self.window_size_samples :]
        self.speaking = False
        self.last_speech_time = None
        self.first_chunk_with_speech = None
        self.current_chunk_idx = 0
