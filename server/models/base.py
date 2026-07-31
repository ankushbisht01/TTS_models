"""
Abstract base class for all TTS model backends.
Every backend must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TTSRequest:
    """Incoming TTS request."""

    text: str
    voice_id: str | None = None  # Name of a saved voice profile
    ref_audio_path: Path | None = None  # Direct path to reference audio
    ref_text: str | None = None  # Transcript of reference audio (if required)
    speed: float = 1.0
    seed: int = -1  # -1 = random
    exaggeration: float = 0.5  # Emotion intensity (if supported)
    voice_description: str | None = None  # Text description of voice (if supported)


@dataclass
class TTSResult:
    """Output from TTS generation."""

    audio: np.ndarray  # Audio waveform as numpy array
    sample_rate: int = 24000
    duration_sec: float = 0.0
    inference_time_sec: float = 0.0
    metadata: dict = field(default_factory=dict)


class TTSBackend(ABC):
    """Abstract base for TTS backends."""

    name: str = "base"
    supports_cloning: bool = False
    supports_streaming: bool = False
    supports_voice_design: bool = False
    supports_emotion_control: bool = False
    supports_lora: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights into GPU memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory."""
        ...

    @abstractmethod
    def generate(self, request: TTSRequest) -> TTSResult:
        """Generate speech from text."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        ...

    def get_info(self) -> dict:
        """Return metadata about this backend."""
        return {
            "name": self.name,
            "loaded": self.is_loaded(),
            "supports_cloning": self.supports_cloning,
            "supports_streaming": self.supports_streaming,
            "supports_voice_design": self.supports_voice_design,
            "supports_emotion_control": self.supports_emotion_control,
            "supports_lora": self.supports_lora,
        }
