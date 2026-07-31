"""
Server configuration — loads from environment variables and .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    workers: int = 1  # Keep at 1 — GPU models aren't fork-safe

    # ── Model Selection ───────────────────────────────────────────────
    # Which backends to load at startup. Comma-separated.
    # Options: f5tts, chatterbox, qwen3tts
    active_backends: str = Field(
        default="f5tts",
        description="Comma-separated list of backends to load (e.g. 'f5tts,chatterbox')",
    )
    default_backend: str = Field(
        default="f5tts",
        description="Default backend when none is specified in the request",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    voices_dir: Path | None = None
    lora_dir: Path | None = None
    models_cache_dir: Path | None = None
    output_dir: Path | None = None

    # ── F5-TTS specific ──────────────────────────────────────────────
    # Config name from f5_tts/configs/*.yaml — NOT the old "F5-TTS" label.
    # Options: F5TTS_v1_Base, F5TTS_v1_Small, F5TTS_Base, F5TTS_Small,
    #          E2TTS_Base, E2TTS_Small
    f5tts_model_type: str = "F5TTS_v1_Base"
    f5tts_ckpt_file: str = ""  # empty = auto-download from HuggingFace
    f5tts_vocab_file: str = ""

    # ── Chatterbox specific ──────────────────────────────────────────
    chatterbox_model: str = "chatterbox"
    chatterbox_exaggeration: float = 0.5  # 0.0 = monotone, 1.0 = very expressive

    # ── Qwen3-TTS specific ───────────────────────────────────────────
    qwen3tts_model: str = "Qwen/Qwen3-TTS"
    qwen3tts_size: str = "0.6B"  # "0.6B" or "1.7B"

    # ── Audio defaults ───────────────────────────────────────────────
    sample_rate: int = 24000
    max_ref_duration_sec: float = 30.0  # Max reference audio length
    max_gen_chars: int = 5000  # Max characters per generation request

    # ── GPU ───────────────────────────────────────────────────────────
    device: str = "cuda"  # "cuda", "cpu", or "cuda:0"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    def model_post_init(self, __context) -> None:
        """Resolve relative paths after init."""
        if self.voices_dir is None:
            self.voices_dir = self.base_dir / "voices"
        if self.lora_dir is None:
            self.lora_dir = self.base_dir / "lora_adapters"
        if self.models_cache_dir is None:
            self.models_cache_dir = self.base_dir / "models_cache"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "output"

        # Ensure directories exist
        for d in [self.voices_dir, self.lora_dir, self.models_cache_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def active_backend_list(self) -> list[str]:
        return [b.strip().lower() for b in self.active_backends.split(",") if b.strip()]


# Singleton
settings = Settings()
