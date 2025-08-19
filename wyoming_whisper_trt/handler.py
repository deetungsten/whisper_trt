"""Event handler for Wyoming Whisper TensorRT server."""

import asyncio
import io
import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

class WhisperTRTEventHandler(AsyncEventHandler):
    """Event handler for Whisper TensorRT."""

    def __init__(
        self,
        wyoming_info: Info,
        model_name: str,
        model_dir: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        data_dir: Optional[Path] = None,
        download_dir: Optional[Path] = None,
    ) -> None:
        """Initialize handler."""
        super().__init__()
        
        self.wyoming_info = wyoming_info
        self.model_name = model_name
        self.model_dir = model_dir
        self.language = language
        self.beam_size = beam_size
        self.data_dir = data_dir
        self.download_dir = download_dir
        
        self._model = None
        self._model_lock = asyncio.Lock()
        
        # Audio processing
        self._audio_buffer = bytes()
        self._sample_rate = 16000
        self._sample_width = 2
        self._channels = 1

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if AudioStart.is_type(event.type):
            # Reset audio buffer
            self._audio_buffer = bytes()
            audio_start = AudioStart.from_event(event)
            self._sample_rate = audio_start.rate
            self._sample_width = audio_start.width
            self._channels = audio_start.channels
            _LOGGER.debug("Audio start: rate=%s, width=%s, channels=%s", 
                         self._sample_rate, self._sample_width, self._channels)
            return True

        if AudioChunk.is_type(event.type):
            # Accumulate audio data
            chunk = AudioChunk.from_event(event)
            self._audio_buffer += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            # Process accumulated audio
            _LOGGER.debug("Audio stop, processing %s bytes", len(self._audio_buffer))
            
            if self._audio_buffer:
                transcript = await self._transcribe_audio(self._audio_buffer)
                await self.write_event(transcript.event())
            
            return True

        if Transcribe.is_type(event.type):
            # Handle direct transcription request
            transcribe = Transcribe.from_event(event)
            if transcribe.audio:
                transcript = await self._transcribe_audio(transcribe.audio)
                await self.write_event(transcript.event())
            
            return True

        return True

    async def _transcribe_audio(self, audio_data: bytes) -> Transcript:
        """Transcribe audio data using Whisper TensorRT."""
        try:
            # Load model if not already loaded
            await self._ensure_model_loaded()
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Write WAV file
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(self._channels)
                    wav_file.setsampwidth(self._sample_width)
                    wav_file.setframerate(self._sample_rate)
                    wav_file.writeframes(audio_data)
                
                temp_path = temp_file.name

            try:
                # Run transcription in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                start_time = time.time()
                
                result = await loop.run_in_executor(
                    None, self._transcribe_file, temp_path
                )
                
                end_time = time.time()
                transcription_time = end_time - start_time
                
                text = result.get("text", "").strip()
                _LOGGER.debug("Transcribed in %.2f seconds: %s", transcription_time, text)
                
                return Transcript(text=text)
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            _LOGGER.exception("Error during transcription: %s", e)
            return Transcript(text="")

    def _transcribe_file(self, audio_file: str) -> dict:
        """Transcribe audio file using Whisper TensorRT model."""
        transcribe_kwargs = {}
        
        if self.language:
            transcribe_kwargs["language"] = self.language
            
        transcribe_kwargs["beam_size"] = self.beam_size
        
        return self._model.transcribe(audio_file, **transcribe_kwargs)

    async def _ensure_model_loaded(self) -> None:
        """Ensure Whisper TensorRT model is loaded."""
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:
                return

            _LOGGER.info("Loading Whisper TensorRT model: %s", self.model_name)
            
            try:
                # Import whisper_trt
                import whisper_trt
                
                # Load model in thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, 
                    self._load_model_sync
                )
                
                _LOGGER.info("Model loaded successfully")
                
            except ImportError:
                _LOGGER.error(
                    "whisper_trt not installed. Install with: pip install whisper-trt"
                )
                raise
            except Exception as e:
                _LOGGER.exception("Failed to load model: %s", e)
                raise

    def _load_model_sync(self):
        """Load Whisper TensorRT model synchronously."""
        import whisper_trt
        
        # Check if model exists in cache
        model_path = self.model_dir / f"{self.model_name}.engine"
        
        if model_path.exists():
            _LOGGER.info("Loading cached TensorRT engine: %s", model_path)
            return whisper_trt.load_trt_model(str(model_path))
        else:
            _LOGGER.info("Building TensorRT engine for model: %s", self.model_name)
            model = whisper_trt.load_trt_model(self.model_name)
            
            # Save the built engine
            model_path.parent.mkdir(parents=True, exist_ok=True)
            _LOGGER.info("Saving TensorRT engine to: %s", model_path)
            # Note: whisper_trt automatically caches engines in ~/.cache/whisper_trt/
            
            return model