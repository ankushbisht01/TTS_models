"""
Voice Cloning & TTS API Server

Provides an OpenAI-compatible /v1/audio/speech endpoint plus voice cloning
management endpoints. Supports multiple TTS backends (F5-TTS, Chatterbox,
Qwen3-TTS) with hot-swappable configuration.
"""

import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import settings
from .models.base import TTSBackend, TTSRequest, TTSResult
from .voice_manager import VoiceManager

# ─── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vc-server")


# ─── Global State ─────────────────────────────────────────────────────

backends: dict[str, TTSBackend] = {}
voice_manager: VoiceManager | None = None


def _load_backend(name: str) -> TTSBackend:
    """Factory function to create and load a backend by name."""
    match name:
        case "f5tts":
            from .models.f5tts_backend import F5TTSBackend

            backend = F5TTSBackend(
                model_type=settings.f5tts_model_type,
                ckpt_file=settings.f5tts_ckpt_file,
                vocab_file=settings.f5tts_vocab_file,
                device=settings.device,
            )
        case "chatterbox":
            from .models.chatterbox_backend import ChatterboxBackend

            backend = ChatterboxBackend(
                device=settings.device,
                default_exaggeration=settings.chatterbox_exaggeration,
            )
        case "qwen3tts":
            from .models.qwen3tts_backend import Qwen3TTSBackend

            backend = Qwen3TTSBackend(
                model_name=settings.qwen3tts_model,
                model_size=settings.qwen3tts_size,
                device=settings.device,
            )
        case _:
            raise ValueError(f"Unknown backend: {name}")

    backend.load()
    return backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global voice_manager

    logger.info("=" * 60)
    logger.info("Voice Cloning Server starting up")
    logger.info("=" * 60)

    # Initialize voice manager
    voice_manager = VoiceManager(settings.voices_dir)
    logger.info("Voice manager: %d profiles loaded", len(voice_manager.list_voices()))

    # Load configured backends
    for backend_name in settings.active_backend_list:
        try:
            logger.info("Loading backend: %s", backend_name)
            backends[backend_name] = _load_backend(backend_name)
            logger.info("✅ Backend loaded: %s", backend_name)
        except ImportError as e:
            logger.error("❌ Backend '%s' not installed: %s", backend_name, e)
        except Exception as e:
            logger.error("❌ Failed to load backend '%s': %s", backend_name, e)

    if not backends:
        logger.warning("⚠️  No backends loaded! Install at least one TTS model.")

    logger.info("Server ready — %d backend(s) active", len(backends))
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down — unloading models...")
    for name, backend in backends.items():
        try:
            backend.unload()
            logger.info("Unloaded: %s", name)
        except Exception as e:
            logger.error("Error unloading %s: %s", name, e)

    backends.clear()


# ─── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Voice Cloning Server",
    description=(
        "Self-hosted voice cloning & TTS server with OpenAI-compatible API. "
        "Supports F5-TTS, Chatterbox, and Qwen3-TTS backends."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response Models ────────────────────────────────────────


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = Field(
        default="f5tts",
        description="Backend to use: 'f5tts', 'chatterbox', 'qwen3tts'",
    )
    input: str = Field(..., description="The text to generate audio for", max_length=5000)
    voice: str = Field(
        default="default",
        description="Voice profile ID to use for cloning",
    )
    response_format: str = Field(default="wav", description="Audio format: 'wav', 'mp3', 'flac'")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speaking speed multiplier")


class VoiceCreateResponse(BaseModel):
    voice_id: str
    name: str
    description: str
    duration_sec: float
    sample_rate: int
    message: str


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: str
    ref_text: str
    duration_sec: float
    preferred_backend: str
    tags: list[str]


class ServerStatus(BaseModel):
    status: str
    backends: dict[str, dict]
    voices_count: int
    gpu_info: dict | None


class GenerateRequest(BaseModel):
    """Extended generation request with full control."""

    text: str = Field(..., max_length=5000)
    backend: str = Field(default="")
    voice_id: str = Field(default="")
    ref_text: str = Field(default="")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    seed: int = Field(default=-1)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    voice_description: str = Field(default="")
    response_format: str = Field(default="wav")


# ─── Helpers ──────────────────────────────────────────────────────────


def _get_backend(name: str = "") -> TTSBackend:
    """Resolve and return a loaded backend."""
    if not backends:
        raise HTTPException(503, "No TTS backends are loaded")

    if not name or name == "tts-1" or name == "tts-1-hd":
        name = settings.default_backend

    if name not in backends:
        available = ", ".join(backends.keys())
        raise HTTPException(
            400, f"Backend '{name}' not available. Loaded backends: {available}"
        )

    return backends[name]


def _audio_to_bytes(audio: np.ndarray, sr: int, fmt: str = "wav") -> bytes:
    """Convert numpy audio array to bytes in the requested format."""
    buf = io.BytesIO()

    match fmt.lower():
        case "wav":
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        case "flac":
            sf.write(buf, audio, sr, format="FLAC")
        case "mp3":
            # soundfile doesn't support mp3 writing natively
            # Fall back to WAV for now
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        case "ogg":
            sf.write(buf, audio, sr, format="OGG", subtype="VORBIS")
        case _:
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")

    buf.seek(0)
    return buf.read()


def _content_type(fmt: str) -> str:
    """Map format to MIME type."""
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(fmt.lower(), "audio/wav")


# ─── API Routes ───────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "backends_loaded": len(backends)}


@app.get("/v1/status", response_model=ServerStatus)
async def server_status():
    """Get server status including loaded backends and GPU info."""
    gpu_info = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                ),
                "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "vram_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            }
    except Exception:
        pass

    return ServerStatus(
        status="ok",
        backends={name: backend.get_info() for name, backend in backends.items()},
        voices_count=len(voice_manager.list_voices()) if voice_manager else 0,
        gpu_info=gpu_info,
    )


# ── OpenAI-Compatible TTS Endpoint ───────────────────────────────────


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible TTS endpoint.

    Drop-in replacement for the OpenAI /v1/audio/speech API.
    Set your base_url to this server and use voice profile IDs as the 'voice' parameter.
    """
    backend = _get_backend(request.model)

    # Resolve voice
    ref_audio_path = None
    ref_text = ""

    if request.voice and request.voice != "default" and voice_manager:
        profile = voice_manager.get_voice(request.voice)
        if profile is None:
            raise HTTPException(404, f"Voice profile '{request.voice}' not found")
        ref_audio_path = voice_manager.get_ref_audio_path(request.voice)
        ref_text = profile.ref_text

    if ref_audio_path is None and backend.supports_cloning:
        raise HTTPException(
            400,
            "No voice profile selected. Create one first via POST /v1/voices/create "
            "or use a voice_id from GET /v1/voices",
        )

    tts_request = TTSRequest(
        text=request.input,
        voice_id=request.voice,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        speed=request.speed,
    )

    try:
        result = backend.generate(tts_request)
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    audio_bytes = _audio_to_bytes(result.audio, result.sample_rate, request.response_format)

    return Response(
        content=audio_bytes,
        media_type=_content_type(request.response_format),
        headers={
            "X-Duration-Sec": str(round(result.duration_sec, 2)),
            "X-Inference-Time-Sec": str(round(result.inference_time_sec, 2)),
            "X-Backend": backend.name,
        },
    )


# ── Extended Generation Endpoint ──────────────────────────────────────


@app.post("/v1/generate")
async def generate_speech(request: GenerateRequest):
    """
    Extended TTS generation with full control over backend, voice, and parameters.
    """
    backend_name = request.backend or settings.default_backend
    backend = _get_backend(backend_name)

    ref_audio_path = None
    ref_text = request.ref_text

    if request.voice_id and voice_manager:
        profile = voice_manager.get_voice(request.voice_id)
        if profile:
            ref_audio_path = voice_manager.get_ref_audio_path(request.voice_id)
            if not ref_text:
                ref_text = profile.ref_text

    tts_request = TTSRequest(
        text=request.text,
        voice_id=request.voice_id,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        speed=request.speed,
        seed=request.seed,
        exaggeration=request.exaggeration,
        voice_description=request.voice_description or None,
    )

    try:
        result = backend.generate(tts_request)
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    audio_bytes = _audio_to_bytes(result.audio, result.sample_rate, request.response_format)

    return Response(
        content=audio_bytes,
        media_type=_content_type(request.response_format),
        headers={
            "X-Duration-Sec": str(round(result.duration_sec, 2)),
            "X-Inference-Time-Sec": str(round(result.inference_time_sec, 2)),
            "X-Backend": backend.name,
        },
    )


# ── Inline Clone: Upload audio + text in one request ─────────────────


@app.post("/v1/clone")
async def clone_inline(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio file (WAV/MP3/FLAC)"),
    ref_text: str = Form(default="", description="Transcript of reference audio"),
    backend: str = Form(default="", description="Backend to use"),
    speed: float = Form(default=1.0),
    exaggeration: float = Form(default=0.5),
    response_format: str = Form(default="wav"),
):
    """
    One-shot voice cloning: upload reference audio and get synthesized speech.
    No need to create a voice profile first.
    """
    backend_name = backend or settings.default_backend
    tts_backend = _get_backend(backend_name)

    # Save uploaded audio to temp file
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await ref_audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        tts_request = TTSRequest(
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            speed=speed,
            exaggeration=exaggeration,
        )

        result = tts_backend.generate(tts_request)

        audio_bytes = _audio_to_bytes(result.audio, result.sample_rate, response_format)

        return Response(
            content=audio_bytes,
            media_type=_content_type(response_format),
            headers={
                "X-Duration-Sec": str(round(result.duration_sec, 2)),
                "X-Inference-Time-Sec": str(round(result.inference_time_sec, 2)),
                "X-Backend": tts_backend.name,
            },
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Voice Profile Management ─────────────────────────────────────────


@app.get("/v1/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List all saved voice profiles."""
    if not voice_manager:
        return []

    return [
        VoiceInfo(
            voice_id=v.voice_id,
            name=v.name,
            description=v.description,
            ref_text=v.ref_text,
            duration_sec=v.duration_sec,
            preferred_backend=v.preferred_backend,
            tags=v.tags,
        )
        for v in voice_manager.list_voices()
    ]


@app.post("/v1/voices/create", response_model=VoiceCreateResponse)
async def create_voice(
    name: str = Form(..., description="Display name for this voice"),
    audio: UploadFile = File(..., description="Reference audio (5-30 seconds, WAV/MP3/FLAC)"),
    ref_text: str = Form(default="", description="Transcript of the reference audio"),
    description: str = Form(default="", description="Description of this voice"),
    preferred_backend: str = Form(default="", description="Preferred TTS backend"),
):
    """
    Create a new voice profile by uploading reference audio.

    The audio should be 5-30 seconds of clean, clear speech from the target speaker.
    Avoid background noise, music, or multiple speakers.
    """
    if not voice_manager:
        raise HTTPException(503, "Voice manager not initialized")

    # Save uploaded file
    suffix = Path(audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        profile = voice_manager.create_voice(
            name=name,
            audio_path=tmp_path,
            ref_text=ref_text,
            description=description,
            preferred_backend=preferred_backend,
        )

        return VoiceCreateResponse(
            voice_id=profile.voice_id,
            name=profile.name,
            description=profile.description,
            duration_sec=profile.duration_sec,
            sample_rate=profile.sample_rate,
            message=f"Voice profile '{profile.name}' created successfully",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile."""
    if not voice_manager:
        raise HTTPException(503, "Voice manager not initialized")

    if voice_manager.delete_voice(voice_id):
        return {"message": f"Voice '{voice_id}' deleted"}
    raise HTTPException(404, f"Voice '{voice_id}' not found")


@app.get("/v1/voices/{voice_id}/audio")
async def get_voice_audio(voice_id: str):
    """Download the reference audio for a voice profile."""
    if not voice_manager:
        raise HTTPException(503, "Voice manager not initialized")

    audio_path = voice_manager.get_ref_audio_path(voice_id)
    if audio_path is None:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    audio_bytes = audio_path.read_bytes()
    return Response(content=audio_bytes, media_type="audio/wav")


# ── Backend Management ────────────────────────────────────────────────


@app.get("/v1/backends")
async def list_backends():
    """List all loaded backends and their capabilities."""
    return {name: backend.get_info() for name, backend in backends.items()}


# ─── Entry Point ──────────────────────────────────────────────────────


def main():
    """CLI entry point for the server."""
    uvicorn.run(
        "server.api_server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
