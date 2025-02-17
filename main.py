import argparse
from math import log
from pathlib import Path
from llangtalk.audio.audio_player import AudioPlayer
from llangtalk.audio.microphone import Microphone
from llangtalk.audio.stream import AudioStream
from llangtalk.asr.huggingface_asr import HuggingfaceASR
from llangtalk.asr.silero_vad import SileroVAD
from llangtalk.llm.ollama_engine import OllamaEngine
import numpy as np
import logging
import asyncio
from llangtalk.rag.FaissSQLiteRAG import FaissSQLiteRAG
from llangtalk.tts.huggingface_tts import HuggingfaceTTS
import re

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="LlamaTalk main script")
parser.add_argument("--input_device_id", type=int, help="Input device id", default=8)
parser.add_argument("--log_level", type=str, help="Log level", default="DEBUG")
parser.add_argument("--asr_device", type=str, help="ASR device", default="cuda:0")
parser.add_argument("--tts_device", type=str, help="TTS device", default="cpu")
parser.add_argument("--llm_device", type=str, help="LLM device", default="cpu")
parser.add_argument("--not_chat_version", action="store_false", help="Use not chat version of LLM")
parser.add_argument("--rag_db_path", type=Path, help="Path to the RAG database", default="rag.db")
parser.add_argument("--rag_folder", type=Path, help="Path to the RAG folder", default="rag")
parser.add_argument(
    "--st_model",
    type=str,
    help="Path to the sentence transformer model",
    default="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
)


def get_audio_for_asr(vad, audio_stream, microphone):
    """
    Processes audio chunks using VAD, identifies speech segments,
    and returns the concatenated audio chunk ready for ASR.
    """
    if vad.POST_SPEECH_TIMEOUT:
        # Calculate the starting chunk index, ensuring it's non-negative
        first_chunk_to_take = vad.first_chunk_with_speech - (0.5 / microphone.BLOCKS_PER_SECOND)
        first_chunk_to_take = int(max(first_chunk_to_take, 0))
        logger.debug(f"Taking chunks from {first_chunk_to_take}")

        # Concatenate chunks from the determined start index
        audio_for_asr = np.concatenate(audio_stream.audio_in_chunks[first_chunk_to_take:])
        logger.debug(f"Audio for ASR: {audio_for_asr.shape}")

        # Reset audio stream and VAD states
        audio_stream.reset()
        vad.soft_reset()

        return audio_for_asr
    return None


async def main(args):

    microphone = Microphone(input_device_index=args.input_device_id)
    asr_model = HuggingfaceASR("openai/whisper-medium", device=args.asr_device)
    audio_stream = AudioStream(microphone.SAMPLING_RATE, target_rate=16000)
    vad = SileroVAD(chunk_size=microphone.BLOCK_SIZE)
    llm = OllamaEngine(chat_version=args.not_chat_version, device=args.llm_device)
    tts = HuggingfaceTTS()
    audio_player = AudioPlayer()
    rag = FaissSQLiteRAG(args.rag_db_path, st_model=args.st_model)
    llm.invoke("Teraz będziemy gadać")
    try:
        while True:
            chunk = microphone.read()
            resampled_chunk = await audio_stream.process_chunk(chunk)
            vad.process_chunk(resampled_chunk)
            audio_for_asr = get_audio_for_asr(vad, audio_stream, microphone)
            if audio_for_asr is not None:
                text = asr_model.predict_audio(audio_for_asr)
                print(f"ASR: {text}")

                print("LLM: ", end="")
                full_text = []
                for chunk in llm.stream(text):
                    full_text.append(chunk)
                    print(chunk, flush=True, end="")
                full_text = " ".join(full_text)
                first_sentence = re.split(r"[\?\!\.]", full_text)[0]
                audio_player.play_audio(*tts.generate_audio_from_text(first_sentence))
    except KeyboardInterrupt:
        logger.info("Stopping application")
    finally:
        microphone.close()


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(main(args))
