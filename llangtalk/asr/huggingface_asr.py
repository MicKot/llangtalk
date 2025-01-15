from llangtalk.asr.asr_interface import ASR
import numpy as np
from transformers import pipeline


class HuggingfaceASR(ASR):
    def __init__(self, model="openai/whisper-small", device="cuda", chunk_length_s=30):

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=chunk_length_s,
            device=device,
        )

    def predict_audio(self, audio: np.ndarray):
        return self.model(inputs=audio)["text"]
