import argparse
from llangtalk.audio.microphone import Microphone
from llangtalk.audio.stream import AudioStream
from llangtalk.asr.huggingface_asr import HuggingfaceASR
from llangtalk.asr.vad_interface import VADEngine
from llangtalk.llm.ollama import Ollama
import numpy as np
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="LlamaTalk main script")
parser.add_argument("--input_device_id", type=int, help="Input device id", default=9)
parser.add_argument("--log_level", type=str, help="Log level", default="DEBUG")
parser.add_argument("--asr_device", type=str, help="ASR device", default="cuda:0")
parser.add_argument("--not_chat_version", action="store_false", help="Use not chat version of LLM")


def main(args):
    microphone = Microphone(input_device_index=args.input_device_id)
    whisper_asr = HuggingfaceASR("openai/whisper-small", device=args.asr_device)
    audio_stream = AudioStream(microphone.SAMPLING_RATE, target_rate=16000)
    vad = VADEngine(chunk_size=microphone.BLOCK_SIZE)
    llm = Ollama(chat_version=args.not_chat_version)
    llm.invoke("Teraz będziemy gadać")
    while True:
        resampled_chunk = audio_stream.process_chunk(microphone.read())
        vad.process_chunk(resampled_chunk)
        if vad.POST_SPEECH_TIMEOUT:
            # we take 0.5s before VAD said that there is speech - it's better for ASR
            first_chunk_to_take = vad.first_chunk_with_speech - (0.5 / microphone.BLOCKS_PER_SECOND)
            first_chunk_to_take = int(max(first_chunk_to_take, 0))
            logger.debug(f"Taking chunks from {first_chunk_to_take}")
            audio_for_asr = np.concatenate(audio_stream.audio_in_chunks[first_chunk_to_take:])
            logger.debug(f"Audio for ASR: {audio_for_asr.shape}")
            text = whisper_asr.predict_audio(audio_for_asr)
            print(f"ASR: {text}")
            audio_stream.reset()
            vad.soft_reset()
            # check if we should stop so normalize text, remove interpunction
            print("LLM: ", end="")
            for chunk in llm.stream(text):
                print(type(chunk))
                print(chunk, flush=True, end="")

    microphone.close()


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)
