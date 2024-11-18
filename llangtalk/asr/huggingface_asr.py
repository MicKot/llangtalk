from numpy import info
from llangtalk.asr.asr_interface import ASREngine
import numpy as np
from transformers import pipeline
import torch


class HuggingfaceASR(ASREngine):
    def __init__(self, model="openai/whisper-small", device="cuda", chunk_length_s=30):

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=chunk_length_s,
            device=device,
        )

    def predict_audio(self, audio: np.ndarray):
        # print(audio)
        return self.model(inputs=audio)["text"]
