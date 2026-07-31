"""
Qwen3-TTS backend — ultra-low latency TTS with voice cloning and voice design.

Apache 2.0 licensed. Supports zero-shot cloning from ~3 seconds of reference
audio AND text-based voice design (create voices from descriptions).
"""

import logging
import time
from pathlib import Path

import numpy as np

from .base import TTSBackend, TTSRequest, TTSResult

logger = logging.getLogger(__name__)


class Qwen3TTSBackend(TTSBackend):
    """Qwen3-TTS: Ultra-low latency TTS with voice cloning and design."""

    name = "qwen3tts"
    supports_cloning = True
    supports_streaming = True
    supports_voice_design = True
    supports_emotion_control = False
    supports_lora = False

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        model_size: str = "0.6B",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.model_size = model_size
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load Qwen3-TTS model into GPU memory."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            )

        model_id = self.model_name
        if self.model_size and self.model_size not in model_id:
            model_id = f"{self.model_name}-{self.model_size}"

        logger.info("Loading Qwen3-TTS model: %s ...", model_id)
        start = time.time()

        try:
            # Try qwen-tts package first (preferred)
            from qwen_tts import QwenTTS

            self._model = QwenTTS(model=model_id, device=self.device)
            self._use_qwen_package = True
        except ImportError:
            # Fallback to direct transformers loading
            logger.info("qwen-tts package not found, using transformers directly")
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._use_qwen_package = False

        elapsed = time.time() - start
        self._loaded = True
        logger.info("Qwen3-TTS loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            del self._tokenizer
            self._tokenizer = None
            self._loaded = False

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-TTS unloaded")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, request: TTSRequest) -> TTSResult:
        """Generate speech using Qwen3-TTS."""
        if not self._loaded:
            raise RuntimeError("Qwen3-TTS model is not loaded. Call load() first.")

        # Qwen3-TTS supports both voice cloning and voice design
        has_ref = request.ref_audio_path is not None and Path(request.ref_audio_path).exists()
        has_description = request.voice_description is not None

        if not has_ref and not has_description:
            raise ValueError(
                "Qwen3-TTS requires either reference audio (ref_audio_path) "
                "or a voice description (voice_description)."
            )

        logger.info(
            "Generating with Qwen3-TTS: text=%d chars, mode=%s",
            len(request.text),
            "clone" if has_ref else "design",
        )

        start = time.time()

        if hasattr(self, "_use_qwen_package") and self._use_qwen_package:
            wav, sr = self._generate_with_package(request, has_ref)
        else:
            wav, sr = self._generate_with_transformers(request, has_ref)

        inference_time = time.time() - start

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
                "mode": "clone" if has_ref else "design",
                "model_size": self.model_size,
            },
        )

    def _generate_with_package(
        self, request: TTSRequest, has_ref: bool
    ) -> tuple[np.ndarray, int]:
        """Generate using the qwen-tts package."""
        kwargs = {"text": request.text}

        if has_ref:
            kwargs["reference_audio"] = str(request.ref_audio_path)
        else:
            kwargs["voice_description"] = request.voice_description

        result = self._model.generate(**kwargs)

        # Handle different return types from the package
        if isinstance(result, tuple):
            return result[0], result[1]
        elif hasattr(result, "audio") and hasattr(result, "sample_rate"):
            return result.audio, result.sample_rate
        else:
            return np.array(result), 24000

    def _generate_with_transformers(
        self, request: TTSRequest, has_ref: bool
    ) -> tuple[np.ndarray, int]:
        """Generate using raw transformers — fallback path."""
        import torch
        import soundfile as sf

        if has_ref:
            ref_audio, ref_sr = sf.read(str(request.ref_audio_path))
            # Build prompt with reference audio embedding
            # This is a simplified version — the full implementation
            # depends on the model's specific chat template
            prompt = f"<|audio|>{request.text}"
        else:
            prompt = (
                f"Voice: {request.voice_description}\n"
                f"Text: {request.text}"
            )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
            )

        # Decode audio tokens — model-specific
        audio_tokens = outputs[0][inputs.input_ids.shape[1] :]
        wav = audio_tokens.float().cpu().numpy()

        return wav, 24000
