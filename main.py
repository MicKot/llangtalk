import argparse
from venv import logger
from hyperpyyaml import load_hyperpyyaml
from llangtalk.asr import whisper_asr
from llangtalk.microphone import Microphone
from llangtalk.asr.whisper_asr import WhisperASR
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
    whisper_asr = WhisperASR()
    data = []
    start = time.perf_counter()
    while time.perf_counter() - start < 5:
        data.extend(microphone.read())
    microphone.close()
    print(whisper_asr.predict_audio(np.array(data)))


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    main(args)
