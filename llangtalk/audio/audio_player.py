import logging
import numpy as np
import simpleaudio as sa

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self):
        pass

    def play_audio(self, audio: np.ndarray, sample_rate: int)
        audio = audio * (2**15 - 1) / np.max(np.abs(audio))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)

        # Start playback
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

        # Wait for playback to finish before exiting
        play_obj.wait_done()