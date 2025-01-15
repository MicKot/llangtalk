import pytest
import numpy as np
from llangtalk.asr.silero_vad import SileroVAD


@pytest.fixture(scope="module")
def vad():
    """Fixture to create a shared SileroVAD instance for all tests."""
    return SileroVAD()


def test_reset_states(vad):
    # Create a dummy audio chunk
    audio_chunk = np.random.rand(16000).astype(np.float32)
    vad.process_chunk(audio_chunk)
    vad.reset_states()
    assert not vad.speaking


def test_soft_reset(vad):
    # Create a dummy audio chunk
    audio_chunk = np.random.rand(16000).astype(np.float32)
    vad.process_chunk(audio_chunk)
    vad.soft_reset()
    assert vad.POST_SPEECH_TIMEOUT == False
    assert vad.speaking == False
    assert vad.last_speech_time == None
    assert vad.first_chunk_with_speech == None
    assert vad.current_chunk_idx == 0
