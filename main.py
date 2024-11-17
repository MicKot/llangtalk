import argparse
from llangtalk.audio.microphone import Microphone
from llangtalk.audio.stream import AudioStream
from llangtalk.asr.huggingface_asr import HuggingfaceASR
from llangtalk.asr.vad_interface import VADEngine
from llangtalk.llm.ollama import Ollama
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="LlamaTalk main script")
parser.add_argument("--input_device_id", type=int, help="Input device id", default=8)
parser.add_argument("--log_level", type=str, help="Log level", default="DEBUG")


def main(args):
    microphone = Microphone(input_device_index=args.input_device_id)
    whisper_asr = HuggingfaceASR()
    audio_stream = AudioStream(microphone.SAMPLING_RATE, target_rate=16000)
    vad = VADEngine()
    llm = Ollama()
    llm.invoke("Teraz będziemy gadać")
    data = []
    start = time.perf_counter()
    while time.perf_counter() - start < 5:
        audio_stream.process_chunk(microphone.read())
        vad.process_chunk(audio_stream.audio)

    microphone.close()
    text = whisper_asr.predict_audio(np.array(data))
    print("ASR:", text)
    for chunk in llm.stream(text):
        print(chunk, flush=True, end="")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)
