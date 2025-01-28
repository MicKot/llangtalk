from queue import Queue
import pyaudio
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def int2float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype("float32")
    if abs_max > 0:
        _sound *= 1 / abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


def callback(input_data, frame_count, time_info, flags, queue=None):
    if queue is not None:
        newsound = np.frombuffer(input_data, np.int16) / 32768
        queue.put(newsound.astype(np.float32))
    return None, pyaudio.paContinue


class Microphone:
    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    SAMPLING_RATE = 44100
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50
    BLOCK_SIZE = 1 / BLOCKS_PER_SECOND

    def __init__(self, input_device_index=9):
        self.data_queue = Queue()
        self.audio = pyaudio.PyAudio()
        self.show_devices()
        self.block_size_input = int(self.SAMPLING_RATE / float(self.BLOCKS_PER_SECOND))

        logger.info("Creating microphone stream")
        logger.debug(
            f"Stream with arguments: input_device_index={input_device_index}, format={self.FORMAT}, channels={self.CHANNELS}, rate={self.SAMPLING_RATE}, input=True, frames_per_buffer={self.block_size_input}"
        )

        self.stream = self.audio.open(
            input_device_index=input_device_index,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLING_RATE,
            input=True,
            frames_per_buffer=self.block_size_input,
            stream_callback=lambda *args: callback(*args, queue=self.data_queue),
        )

        self.start()

    def start(self):
        self.stream.start_stream()

    def read(self):
        return self.data_queue.get()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def show_devices(self):
        # if logger.level == logging.DEBUG:
        logger.debug("Show devices")
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        for i in range(0, numdevices):
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")) > 0:
                print(
                    "Input Device id ",
                    i,
                    " - ",
                    self.audio.get_device_info_by_host_api_device_index(0, i).get("name"),
                )
