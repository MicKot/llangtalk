import numpy as np
from scipy.signal import resample


class AudioStream:
    def __init__(self, original_rate: int, target_rate: int):
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.audio = np.array([], dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray):
        """
        Process an audio chunk by resampling it and appending it to the audio stream.

        Parameters:
        - chunk: A numpy array representing the audio data.
        """
        resampled = self.resample_chunk(chunk)
        self.audio = np.append(self.audio, resampled)
        return resampled

    def resample_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Resample an audio chunk to the target sample rate.

        Parameters:
        - chunk: A numpy array representing the audio data.

        Returns:
        - A numpy array of the resampled audio data.
        """
        num_samples = int(len(chunk) * self.target_rate / self.original_rate)
        resampled_chunk = resample(chunk, num_samples)
        return resampled_chunk
