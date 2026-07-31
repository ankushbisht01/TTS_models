"""
Chatterbox backend — production-grade voice cloning with emotion control.

MIT licensed. Supports zero-shot cloning from ~5 seconds of reference audio,
paralinguistic tags ([laugh], [sigh]), and emotion exaggeration control.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from .base import TTSBackend, TTSRequest, TTSResult

logger = logging.getLogger(__name__)


class ChatterboxBackend(TTSBackend):
    """Chatterbox: Production TTS with emotion control and watermarking."""

    name = "chatterbox"
    supports_cloning = True
    supports_streaming = False
    supports_voice_design = False
    supports_emotion_control = True
    supports_lora = False

    def __init__(
        self,
        model_name: str = "chatterbox",
        device: str = "cuda",
        default_exaggeration: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device
        self.default_exaggeration = default_exaggeration
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load Chatterbox model into GPU memory."""
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "Chatterbox is not installed. Run: pip install -e '.[chatterbox]' "
                "or: pip install chatterbox-tts"
            )

        logger.info("Loading Chatterbox model...")
        start = time.time()

        self._model = ChatterboxTTS.from_pretrained(device=self.device)

        elapsed = time.time() - start
        self._loaded = True
        logger.info("Chatterbox loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Chatterbox unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Chatterbox with voice cloning and emotion control."""
        if not self._loaded:
            raise RuntimeError("Chatterbox model is not loaded. Call load() first.")

        if request.ref_audio_path is None:
            raise ValueError(
                "Chatterbox requires reference audio for voice cloning. "
                "Provide ref_audio_path in the request."
            )

        ref_path = Path(request.ref_audio_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        exaggeration = request.exaggeration if request.exaggeration is not None else self.default_exaggeration

        logger.info(
            "Generating with Chatterbox: text=%d chars, ref=%s, exaggeration=%.2f",
            len(request.text),
            ref_path.name,
            exaggeration,
        )

        start = time.time()

        wav = self._model.generate(
            text=request.text,
            audio_prompt_path=str(ref_path),
            exaggeration=exaggeration,
        )

        inference_time = time.time() - start

        # Convert to numpy if tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        if wav.ndim > 1:
            wav = wav.squeeze()

        sr = self._model.sr if hasattr(self._model, "sr") else 24000
        duration = len(wav) / sr

        logger.info(
            "Chatterbox generated %.1fs audio in %.2fs (RTF=%.2f)",
            duration,
            inference_time,
            inference_time / duration if duration > 0 else 0,
        )

        return TTSResult(
            audio=wav,
            sample_rate=sr,
            duration_sec=duration,
            inference_time_sec=inference_time,
            metadata={
                "backend": self.name,
                "ref_audio": ref_path.name,
                "exaggeration": exaggeration,
            },
        )
