from faster_whisper import WhisperModel
from numpy import info
from llangtalk.asr.asr_interface import ASREngine
import numpy as np


class WhisperASR(ASREngine):
    def __init__(self):
        model_size = "small"

        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def predict_audio(self, audio: np.ndarray):
        segment, info = self.model.transcribe(audio)
        return " ".join(segment.text for segment in segment)
