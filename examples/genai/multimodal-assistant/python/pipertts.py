"""
PiperTTS - A wrapper class for Piper TTS model to synthesize speech from text.

This module provides an interface to load a Piper ONNX model and generate speech audio:
- Supports WAV audio generation as a memory buffer
- Encodes audio as Base64 for WebSocket or network transmission
- Saves generated audio to disk as a standard WAV file

Dependencies:
-------------
- piper-tts >= 1.3.0
- Python standard libraries: io, wave, base64, os, tempfile
"""

import base64
import io
import os
import tempfile
import wave
from piper import PiperVoice, SynthesisConfig

class PiperTTS:
    """
    PiperTTS class for text-to-speech synthesis using Piper ONNX models.
    Provides methods to synthesize speech as audio buffers, base64-encoded audio, or WAV files.
    """

    DEFAULT_MODEL_PATH = "assets/en_US-amy-low.onnx"
    DEFAULT_SAMPLE_RATE = 22050

    def __init__(self, model_path=None, sample_rate=DEFAULT_SAMPLE_RATE, use_cuda=False, config=None):
        """
        Initialize the Piper TTS engine.

        Args:
            model_path (str, optional): Path to the Piper ONNX model file.
            sample_rate (int): Desired sample rate in Hz (used for playback/export). Actual model rate may differ.
            use_cuda (bool): Whether to use GPU (requires onnxruntime-gpu).
            config (SynthesisConfig, optional): Custom synthesis config.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.sample_rate = sample_rate
        self.voice = PiperVoice.load(self.model_path, use_cuda=use_cuda)

        self.config = config or SynthesisConfig(
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
            normalize_audio=True
        )

    def synthesize(self, text):
        """
        Synthesize speech from the provided text into a WAV audio buffer.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmp_path = tmpfile.name

        # Now open the file with `wave.open()` to get a Wave_write object
        with wave.open(tmp_path, 'wb') as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.config)

        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()

        os.remove(tmp_path)
        return io.BytesIO(audio_bytes)

    def synthesize_base64(self, text):
        """
        Synthesize speech and encode the audio as a Base64 string.

        Args:
            text (str): Input text to synthesize.

        Returns:
            str: Base64-encoded WAV audio.
        """
        buffer = self.synthesize(text)
        b64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        return b64_audio

    def save_audio(self, buffer, filename):
        """
        Save the synthesized audio buffer to a WAV file on disk.

        Args:
            buffer (io.BytesIO): Audio buffer containing WAV data.
            filename (str): Output file path.
        """
        with open(filename, 'wb') as f:
            buffer.seek(0)
            f.write(buffer.read())
        print(f"Saved audio to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Piper TTS module.")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file path")

    args = parser.parse_args()

    print(f"🔊 Synthesizing: \"{args.text}\"")
    tts = PiperTTS()
    buffer = tts.synthesize(args.text)
    tts.save_audio(buffer, args.output)