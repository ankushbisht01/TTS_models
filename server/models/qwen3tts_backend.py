"""
Qwen3-TTS backend — low-latency TTS with voice cloning and voice design.

Apache 2.0 licensed. Three model variants, each exposing a different generation
mode (the variant is inferred from the model id):

  *-Base         zero-shot cloning from reference audio  -> generate_voice_clone
  *-CustomVoice  named preset speakers                   -> generate_custom_voice
  *-VoiceDesign  voice described in natural language     -> generate_voice_design

Use a *-Base checkpoint to work with the /v1/voices profiles, since that is the
only variant that accepts reference audio.
"""

import logging
import time
from pathlib import Path

import numpy as np

from .base import TTSBackend, TTSRequest, TTSResult

logger = logging.getLogger(__name__)

# Generation mode by model-id suffix.
_MODE_BASE = "clone"
_MODE_CUSTOM = "custom"
_MODE_DESIGN = "design"

_SUFFIX_MODES = {
    "-base": _MODE_BASE,
    "-customvoice": _MODE_CUSTOM,
    "-voicedesign": _MODE_DESIGN,
}

_KNOWN_MODELS = (
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)


class Qwen3TTSBackend(TTSBackend):
    """Qwen3-TTS: low-latency TTS with cloning, preset speakers and voice design."""

    name = "qwen3tts"
    supports_cloning = True
    supports_streaming = True
    supports_voice_design = True
    supports_emotion_control = False
    supports_lora = False

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda",
        dtype: str = "bfloat16",
        language: str = "English",
        speaker: str = "",
        instruct: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.language = language
        self.speaker = speaker
        self.instruct = instruct

        self.mode = self._infer_mode(model_name)
        # Only the Base checkpoints accept reference audio.
        self.supports_cloning = self.mode == _MODE_BASE

        self._model = None
        self._loaded = False

    @staticmethod
    def _infer_mode(model_name: str) -> str:
        lowered = model_name.lower()
        for suffix, mode in _SUFFIX_MODES.items():
            if lowered.endswith(suffix):
                return mode
        raise ValueError(
            f"Cannot determine generation mode from Qwen3-TTS model '{model_name}'. "
            f"The id must end in -Base, -CustomVoice or -VoiceDesign. "
            f"Known models: {', '.join(_KNOWN_MODELS)}"
        )

    def load(self) -> None:
        """Load Qwen3-TTS model into GPU memory."""
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "Qwen3-TTS is not installed. Run: pip install -e '.[qwen3tts]' "
                "or: pip install qwen-tts"
            )

        import torch

        logger.info("Loading Qwen3-TTS model: %s (mode=%s)...", self.model_name, self.mode)
        start = time.time()

        # NB: no attn_implementation="flash_attention_2" — the model card
        # suggests it, but flash-attn needs a separate compiled build. The
        # default (sdpa) works everywhere and is fast enough here.
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=getattr(torch, self.dtype, torch.bfloat16),
        )

        elapsed = time.time() - start
        self._loaded = True
        logger.info("Qwen3-TTS loaded in %.1fs", elapsed)

        try:
            langs = self._model.get_supported_languages()
            if langs and self.language not in langs:
                logger.warning(
                    "Language '%s' not in this model's supported set: %s",
                    self.language,
                    ", ".join(langs),
                )
            if self.mode == _MODE_CUSTOM:
                speakers = self._model.get_supported_speakers()
                logger.info("Supported speakers: %s", ", ".join(speakers or []))
                if speakers and self.speaker not in speakers:
                    logger.warning(
                        "Speaker '%s' not supported by this model; generation will fail. "
                        "Set QWEN3TTS_SPEAKER to one of the above.",
                        self.speaker,
                    )
        except Exception as e:  # non-fatal introspection
            logger.debug("Could not query model capabilities: %s", e)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-TTS unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, request: TTSRequest) -> TTSResult:
        """Generate speech using whichever mode this checkpoint supports."""
        if not self._loaded:
            raise RuntimeError("Qwen3-TTS model is not loaded. Call load() first.")

        language = self.language
        instruct = request.voice_description or self.instruct

        start = time.time()

        if self.mode == _MODE_BASE:
            ref_path = request.ref_audio_path
            if ref_path is None or not Path(ref_path).exists():
                raise ValueError(
                    f"{self.model_name} clones from reference audio — provide a voice "
                    "profile (POST /v1/voices/create) or switch to a -CustomVoice / "
                    "-VoiceDesign checkpoint."
                )
            if not request.ref_text:
                # Without a transcript the model falls back to speaker-embedding
                # only, which clones timbre but not prosody.
                logger.warning(
                    "No ref_text for '%s' — using x-vector-only mode (lower fidelity)",
                    request.voice_id,
                )
            wavs, sr = self._model.generate_voice_clone(
                text=request.text,
                language=language,
                ref_audio=str(ref_path),
                ref_text=request.ref_text or None,
                x_vector_only_mode=not bool(request.ref_text),
            )

        elif self.mode == _MODE_CUSTOM:
            wavs, sr = self._model.generate_custom_voice(
                text=request.text,
                speaker=self.speaker,
                language=language,
                instruct=instruct or None,
            )

        else:  # _MODE_DESIGN
            if not instruct:
                raise ValueError(
                    f"{self.model_name} designs a voice from a description — set "
                    "voice_description on the request or QWEN3TTS_INSTRUCT in .env."
                )
            wavs, sr = self._model.generate_voice_design(
                text=request.text,
                instruct=instruct,
                language=language,
            )

        inference_time = time.time() - start

        # Every generate_* returns (List[np.ndarray], sample_rate)
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        if wav.ndim > 1:
            wav = wav.squeeze()

        duration = len(wav) / sr

        logger.info(
            "Qwen3-TTS generated %.1fs audio in %.2fs (RTF=%.2f)",
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
                "model": self.model_name,
                "mode": self.mode,
                "language": language,
            },
        )
