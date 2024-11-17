import torch
import numpy as np


class VADEngine:
    def __init__(self, sample_rate=16000):
        self.model, self.utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
        self.sample_rate = sample_rate
        self.internal_audio_buffer = []

    def process_chunk(self, chunk: np.ndarray):
        pass
