"""
PiperTTS - A wrapper class for Piper TTS model to synthesize speech from text.

This module provides an interface to load a Piper ONNX model and generate speech audio:
- Supports WAV audio generation as a memory buffer
- Streams audio in chunks so playback can start before synthesis finishes, or
  synthesizes whole sentences for voices without an encoder/decoder pair
- Encodes audio as Base64 for WebSocket or network transmission
- Saves generated audio to disk as a standard WAV file

Dependencies:
-------------
- piper-tts >= 1.3.0
- onnxruntime
- numpy
- Python standard libraries: io, wave, base64, os, tempfile
"""

import base64
import io
import os
import tempfile
import wave
import numpy as np
import onnxruntime
from piper import PiperVoice, SynthesisConfig
from piper.voice import AudioChunk

class PiperTTS:
    """
    PiperTTS class for text-to-speech synthesis using Piper ONNX models.
    Provides methods to synthesize speech as audio buffers, base64-encoded audio, or WAV files.
    """

    DEFAULT_MODEL_PATH = "assets/en_US-amy-low.onnx"
    DEFAULT_SAMPLE_RATE = 22050
    DEFAULT_NUM_THREADS = 8   # ONNX Runtime's own default oversubscribes this CPU
    CHUNK_FRAMES = 45         # latent frames per streaming step, about 0.5 s of audio
    CHUNK_PADDING = 10        # context frames decoded either side, then trimmed off

    def __init__(self, model_path=None, sample_rate=DEFAULT_SAMPLE_RATE, config=None, streaming=True):
        """
        Initialize the Piper TTS engine.

        Args:
            model_path (str, optional): Path to the Piper ONNX model file.
            sample_rate (int): Desired sample rate in Hz (used for playback/export). Actual model rate may differ.
            config (SynthesisConfig, optional): Custom synthesis config.
            streaming (bool): Decode the sentence in slices so playback can start early.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.sample_rate = sample_rate
        self.voice = PiperVoice.load(self.model_path)

        self.config = config or SynthesisConfig(
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
            normalize_audio=True
        )

        base_path = os.path.splitext(self.model_path)[0]
        encoder_path = f"{base_path}.enc.onnx"
        decoder_path = f"{base_path}.dec.onnx"
        self.streaming = (streaming and os.path.exists(encoder_path)
                          and os.path.exists(decoder_path))
        if self.streaming:
            self.enc_session = self._make_session(encoder_path)
            self.dec_session = self._make_session(decoder_path)
            self.dec_input_name = self.dec_session.get_inputs()[0].name

            # Trial run: warms both sessions and gives the samples-per-frame ratio
            phoneme_ids = self.voice.phonemes_to_ids(self.voice.phonemize("Ready.")[0])
            latent = self.enc_session.run(None, self._encoder_args(phoneme_ids))[0]
            audio = self.dec_session.run(None, {self.dec_input_name: latent})[0].squeeze()
            self.upsample_factor = round(audio.shape[0] / latent.shape[2])

    def _make_session(self, model_path):
        """Create an ONNX Runtime session with the thread count tuned for this CPU."""
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.DEFAULT_NUM_THREADS
        return onnxruntime.InferenceSession(model_path, sess_options=options,
                                            providers=["CPUExecutionProvider"])

    def _encoder_args(self, phoneme_ids):
        """Build the encoder inputs, mirroring PiperVoice.phoneme_ids_to_audio."""
        return {
            "input": np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0),
            "input_lengths": np.array([len(phoneme_ids)], dtype=np.int64),
            "scales": np.array([self.config.noise_scale, self.config.length_scale,
                                self.config.noise_w_scale], dtype=np.float32),
        }


    def set_utterance_speed(self, speed):
        """Set speech speed as a multiplier where 1.0 is normal."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))
        self.config.length_scale = 1.0 / speed

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

    def synthesize_stream(self, text):
        """
        Synthesize speech and yield a WAV buffer per piece as soon as it is ready,
        or the whole utterance as a single buffer when not streaming.
        """
        if not self.streaming:
            yield self.synthesize(text)
            return

        for phonemes in self.voice.phonemize(text):
            phoneme_ids = self.voice.phonemes_to_ids(phonemes)
            latent = self.enc_session.run(None, self._encoder_args(phoneme_ids))[0]
            total_frames = latent.shape[2]

            start = 0
            while start < total_frames:
                end = min(start + self.CHUNK_FRAMES, total_frames)
                padded_start = max(0, start - self.CHUNK_PADDING)
                padded_end = min(total_frames, end + self.CHUNK_PADDING)

                window = np.ascontiguousarray(latent[:, :, padded_start:padded_end])
                audio = self.dec_session.run(None, {self.dec_input_name: window})[0].squeeze()

                # Drop the padding; a relative -0 tail index would slice the last chunk away.
                head = (start - padded_start) * self.upsample_factor
                tail = audio.shape[0] - (padded_end - end) * self.upsample_factor
                chunk = AudioChunk(self.voice.config.sample_rate, 2, 1, audio[head:tail])

                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                    wav_file.writeframes(chunk.audio_int16_bytes)
                yield buffer

                start = end

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