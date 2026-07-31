"""
F5-TTS backend — flow-matching voice cloning with optional LoRA fine-tuning.

This is the recommended primary backend for highest-quality voice cloning.
Supports zero-shot cloning from 5-15 seconds of reference audio.
"""

import logging
import time
from pathlib import Path

import numpy as np

from .base import TTSBackend, TTSRequest, TTSResult

logger = logging.getLogger(__name__)

# f5-tts >=1.0 selects a model by config name (f5_tts/configs/*.yaml) via the
# `model` kwarg. Older releases used `model_type` with these friendly names, so
# translate them rather than breaking existing .env files.
_LEGACY_MODEL_NAMES = {
    "F5-TTS": "F5TTS_v1_Base",
    "F5TTS": "F5TTS_v1_Base",
    "E2-TTS": "E2TTS_Base",
    "E2TTS": "E2TTS_Base",
}

_KNOWN_MODELS = {
    "F5TTS_v1_Base",
    "F5TTS_v1_Small",
    "F5TTS_Base",
    "F5TTS_Small",
    "E2TTS_Base",
    "E2TTS_Small",
}


class F5TTSBackend(TTSBackend):
    """F5-TTS: Flow-matching based text-to-speech with voice cloning."""

    name = "f5tts"
    supports_cloning = True
    supports_streaming = False
    supports_voice_design = False
    supports_emotion_control = False
    supports_lora = True

    def __init__(
        self,
        model_type: str = "F5TTS_v1_Base",
        ckpt_file: str = "",
        vocab_file: str = "",
        device: str = "cuda",
        hf_cache_dir: str | None = None,
    ):
        self.model_type = _LEGACY_MODEL_NAMES.get(model_type, model_type)
        if self.model_type != model_type:
            logger.warning(
                "F5-TTS model name '%s' is from the pre-1.0 API; using '%s'. "
                "Set F5TTS_MODEL_TYPE=%s to silence this.",
                model_type,
                self.model_type,
                self.model_type,
            )
        if self.model_type not in _KNOWN_MODELS:
            raise ValueError(
                f"Unknown F5-TTS model '{self.model_type}'. "
                f"Expected one of: {', '.join(sorted(_KNOWN_MODELS))}"
            )

        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load F5-TTS model into GPU memory."""
        try:
            from f5_tts.api import F5TTS
        except ImportError:
            raise ImportError(
                "F5-TTS is not installed. Run: pip install -e '.[f5tts]' "
                "or: pip install f5-tts"
            )

        logger.info("Loading F5-TTS model (%s)...", self.model_type)
        start = time.time()

        # NB: the kwarg is `model`, not `model_type` — it was renamed in
        # f5-tts 1.0. Weights auto-download from HuggingFace on first load.
        kwargs = {
            "model": self.model_type,
            "ckpt_file": self.ckpt_file or "",
            "vocab_file": self.vocab_file or "",
            "device": self.device,
        }
        if self.hf_cache_dir:
            kwargs["hf_cache_dir"] = self.hf_cache_dir

        self._model = F5TTS(**kwargs)

        elapsed = time.time() - start
        self._loaded = True
        logger.info("F5-TTS loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("F5-TTS unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, request: TTSRequest) -> TTSResult:
        """Generate speech using F5-TTS with voice cloning."""
        if not self._loaded:
            raise RuntimeError("F5-TTS model is not loaded. Call load() first.")

        if request.ref_audio_path is None:
            raise ValueError(
                "F5-TTS requires reference audio for voice cloning. "
                "Provide ref_audio_path in the request."
            )

        ref_path = Path(request.ref_audio_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        logger.info(
            "Generating with F5-TTS: text=%d chars, ref=%s",
            len(request.text),
            ref_path.name,
        )

        start = time.time()

        # F5-TTS infer returns (wav_numpy, sample_rate, spectrogram).
        # seed must be None for "random" — passing -1 pins the RNG and makes
        # every generation byte-identical.
        wav, sr, _ = self._model.infer(
            ref_file=str(ref_path),
            ref_text=request.ref_text or "",
            gen_text=request.text,
            seed=request.seed if request.seed >= 0 else None,
            speed=request.speed,
        )

        inference_time = time.time() - start

        # Ensure numpy array
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        # Flatten if needed
        if wav.ndim > 1:
            wav = wav.squeeze()

        duration = len(wav) / sr

        logger.info(
            "F5-TTS generated %.1fs audio in %.2fs (RTF=%.2f)",
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
                "model_type": self.model_type,
                "ref_audio": ref_path.name,
                # The model records the seed it actually used when we pass None
                "seed": getattr(self._model, "seed", request.seed),
            },
        )
