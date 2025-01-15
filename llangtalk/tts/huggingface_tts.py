from llangtalk.tts.tts_interface import TTSEngine
from transformers import pipeline
from datasets import load_dataset
import torch


class HuggingfaceTTS(TTSEngine):
    def __init__(self, model_name="d190305/speecht5_finetuned_voxpopuli_pl_full_dataset", device="cpu"):
        self.device = device
        self.synthesiser = pipeline("text-to-speech", model_name, device="cpu")

        if "t5" in model_name:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embedding = torch.tensor(embeddings_dataset[1301]["xvector"]).unsqueeze(0)
            self.sample_rate = 22050

    def generate_audio_from_text(self, text: str = "chłopczyk, co u ciebie?"):
        speech = self.synthesiser(text, forward_params={"speaker_embeddings": self.speaker_embedding})
        return speech["audio"]
