"""
Real-Time STT Tester with Silero VAD
=====================================
Only transcribes when speech is detected - no gibberish!

Usage:
    python test_stt.py              # Interactive menu
    python test_stt.py groq         # Test specific model
    python test_stt.py --list       # List available models
"""

import sys
import io
import os
import wave
import threading
import queue
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# VAD settings from .env
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25"))
VAD_MIN_SILENCE_DURATION = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.3"))


class SileroVAD:
    """Silero VAD wrapper for real-time speech detection."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad()
            print("✅ Silero VAD loaded")
        except ImportError:
            print("⚠️  Silero VAD not installed. Install with: pip install silero-vad torch torchaudio")
            print("⚠️  Running without VAD (will transcribe everything)")
            self.model = None
    
    def reset_states(self):
        """Reset VAD internal states."""
        if self.model is not None:
            self.model.reset_states()
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        
        Args:
            audio_chunk: numpy array of int16 audio samples
            
        Returns:
            True if speech detected, False otherwise
        """
        if self.model is None:
            return True  # No VAD, assume all is speech
        
        import torch
        
        # Convert int16 to float32 normalized [-1, 1]
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float)
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
        
        return speech_prob >= self.threshold


class RealTimeTranscriber:
    """Real-time speech transcription with VAD."""
    
    def __init__(self, model_id: str):
        from stt_config import get_model
        
        self.name, self.api_key, self.transcribe_fn = get_model(model_id)
        
        # Initialize VAD
        self.vad = SileroVAD(threshold=VAD_THRESHOLD)
        
        # Audio buffer for accumulating speech
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        
        # Frame settings (512 samples = 32ms at 16kHz, required by Silero)
        self.frame_size = 512
        self.min_speech_frames = int(VAD_MIN_SPEECH_DURATION * SAMPLE_RATE / self.frame_size)
        self.min_silence_frames = int(VAD_MIN_SILENCE_DURATION * SAMPLE_RATE / self.frame_size)
        
        self.speech_frames_count = 0
        self.running = False
    
    def _audio_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio to WAV bytes."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        return wav_buffer.getvalue()
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single audio frame with VAD."""
        is_speech = self.vad.is_speech(frame)
        
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames = 0
            
            # Start accumulating speech
            if not self.is_speaking and self.speech_frames_count >= self.min_speech_frames:
                self.is_speaking = True
                print("🎤 ", end="", flush=True)
            
            if self.is_speaking:
                self.speech_buffer.append(frame)
        
        else:
            self.speech_frames_count = 0
            
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_buffer.append(frame)  # Include trailing silence
                
                # End of speech detected
                if self.silence_frames >= self.min_silence_frames:
                    self._transcribe_buffer()
                    self.is_speaking = False
                    self.speech_buffer = []
                    self.silence_frames = 0
                    self.vad.reset_states()
    
    def _transcribe_buffer(self):
        """Transcribe accumulated speech buffer."""
        if not self.speech_buffer:
            return
        
        # Combine all speech frames
        audio_data = np.concatenate(self.speech_buffer, axis=0)
        wav_bytes = self._audio_to_wav_bytes(audio_data)
        
        duration = len(audio_data) / SAMPLE_RATE
        print(f"({duration:.1f}s) ", end="", flush=True)
        
        try:
            text = self.transcribe_fn(wav_bytes, self.api_key, SAMPLE_RATE)
            if text and text.strip():
                print(f"→ {text}")
            else:
                print("→ (no speech)")
        except Exception as e:
            print(f"→ ❌ {str(e)[:50]}")
    
    def start(self):
        """Start real-time transcription."""
        import sounddevice as sd
        
        print(f"\n{'=' * 60}")
        print(f"🎙️  REAL-TIME STT: {self.name}")
        print(f"{'=' * 60}")
        print(f"📊 VAD Threshold: {VAD_THRESHOLD}")
        print(f"📊 Min Speech: {VAD_MIN_SPEECH_DURATION}s | Min Silence: {VAD_MIN_SILENCE_DURATION}s")
        print(f"{'─' * 60}")
        print("🔴 Listening... (Ctrl+C to stop)\n")
        
        self.running = True
        audio_queue = queue.Queue()
        
        def callback(indata, frames, time_info, status):
            if self.running:
                audio_queue.put(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.int16,
                callback=callback,
                blocksize=self.frame_size
            ):
                while self.running:
                    try:
                        frame = audio_queue.get(timeout=0.1)
                        self._process_frame(frame.flatten())
                    except queue.Empty:
                        continue
                        
        except KeyboardInterrupt:
            self.running = False
            print(f"\n{'─' * 60}")
            print("🛑 Stopped.")


def list_models():
    """List all configured models."""
    from stt_config import STT_MODELS, get_available_models
    
    print("\n" + "=" * 60)
    print("📋 AVAILABLE STT MODELS")
    print("=" * 60)
    
    available = get_available_models()
    available_ids = [m[0] for m in available]
    
    for model_id, (name, env_key, _) in STT_MODELS.items():
        status = "✅ Ready" if model_id in available_ids else "❌ No API key"
        print(f"  {model_id:15} | {name:25} | {status}")
    
    print("=" * 60)
    return available


def select_model():
    """Interactive model selection."""
    available = list_models()
    
    if not available:
        print("\n❌ No models configured! Add API keys to .env file")
        sys.exit(1)
    
    print("\nEnter model ID to test (or 'q' to quit):")
    
    while True:
        choice = input("> ").strip().lower()
        if choice == 'q':
            sys.exit(0)
        if choice in [m[0] for m in available]:
            return choice
        print(f"Invalid. Available: {[m[0] for m in available]}")


def main():
    print("\n" + "=" * 60)
    print("🎙️  REAL-TIME STT TESTER (with Silero VAD)")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ("--list", "-l"):
            list_models()
            return
        
        if arg in ("--help", "-h"):
            print(__doc__)
            return
        
        # Run with specified model
        transcriber = RealTimeTranscriber(arg)
        transcriber.start()
    else:
        model_id = select_model()
        transcriber = RealTimeTranscriber(model_id)
        transcriber.start()


if __name__ == "__main__":
    main()
