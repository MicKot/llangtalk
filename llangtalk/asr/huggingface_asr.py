from numpy import info
from llangtalk.asr.asr_interface import ASREngine
import numpy as np
from transformers import pipeline
import torchaudio
import torch


class HuggingfaceASR(ASREngine):
    def __init__(self, model="openai/whisper-small", device="cuda"):

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            device=device,
        )

    def predict_audio(self, audio: np.ndarray):
        return self.model(
            torchaudio.functional.resample(
                waveform=torch.tensor(audio),
                orig_freq=44100,
                new_freq=16000,
            ).numpy()
        )["text"]
