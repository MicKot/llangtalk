import logging
import numpy as np
import simpleaudio as sa

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self):
        pass

    @staticmethod
    def play_audio(audio: np.ndarray, sample_rate: int):
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio * (2**15 - 1) / np.max(np.abs(audio))
            audio = audio.astype(np.int16)
        # Start playback
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

        # Wait for playback to finish before exiting
        play_obj.wait_done()
