"""
API endpoint tests for the Voice Cloning Server.

Run with: pytest tests/test_api.py -v
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings():
    """Override settings for testing."""
    with patch("server.config.Settings") as mock:
        s = mock.return_value
        s.active_backends = ""  # Don't load real models
        s.default_backend = "f5tts"
        s.active_backend_list = []
        s.voices_dir = Path("/tmp/test_voices")
        s.lora_dir = Path("/tmp/test_lora")
        s.models_cache_dir = Path("/tmp/test_models")
        s.output_dir = Path("/tmp/test_output")
        s.log_level = "warning"
        s.host = "0.0.0.0"
        s.port = 8000
        s.workers = 1
        s.device = "cpu"
        yield s


@pytest.fixture
def client():
    """Create a test client without loading real models."""
    from server.api_server import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "backends_loaded" in data

    def test_server_status(self, client):
        response = client.get("/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "backends" in data
        assert "voices_count" in data


class TestVoiceEndpoints:
    """Test voice profile management."""

    def test_list_voices_empty(self, client):
        response = client.get("/v1/voices")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_backends(self, client):
        response = client.get("/v1/backends")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)


class TestAudioHelpers:
    """Test audio conversion utilities."""

    def test_audio_to_bytes_wav(self):
        from server.api_server import _audio_to_bytes

        audio = np.random.randn(24000).astype(np.float32) * 0.5
        result = _audio_to_bytes(audio, 24000, "wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify it's valid WAV
        buf = io.BytesIO(result)
        data, sr = sf.read(buf)
        assert sr == 24000
        assert len(data) == 24000

    def test_audio_to_bytes_flac(self):
        from server.api_server import _audio_to_bytes

        audio = np.random.randn(24000).astype(np.float32) * 0.5
        result = _audio_to_bytes(audio, 24000, "flac")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_content_type_mapping(self):
        from server.api_server import _content_type

        assert _content_type("wav") == "audio/wav"
        assert _content_type("mp3") == "audio/mpeg"
        assert _content_type("flac") == "audio/flac"
        assert _content_type("ogg") == "audio/ogg"
        assert _content_type("unknown") == "audio/wav"


class TestVoiceManager:
    """Test voice profile manager."""

    def test_create_and_list(self, tmp_path):
        from server.voice_manager import VoiceManager

        manager = VoiceManager(tmp_path / "voices")

        # Create test audio
        audio_path = tmp_path / "test_ref.wav"
        audio = np.random.randn(48000).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio, 24000)

        # Create voice
        profile = manager.create_voice(
            name="Test Voice",
            audio_path=audio_path,
            ref_text="This is a test",
            description="Test voice for unit tests",
        )

        assert profile.voice_id == "test_voice"
        assert profile.name == "Test Voice"
        assert profile.duration_sec > 0

        # List voices
        voices = manager.list_voices()
        assert len(voices) == 1
        assert voices[0].voice_id == "test_voice"

        # Get voice
        v = manager.get_voice("test_voice")
        assert v is not None
        assert v.name == "Test Voice"

        # Get audio path
        audio = manager.get_ref_audio_path("test_voice")
        assert audio is not None
        assert audio.exists()

    def test_delete_voice(self, tmp_path):
        from server.voice_manager import VoiceManager

        manager = VoiceManager(tmp_path / "voices")

        audio_path = tmp_path / "test_ref.wav"
        audio = np.random.randn(48000).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio, 24000)

        manager.create_voice(name="Delete Me", audio_path=audio_path)
        assert len(manager.list_voices()) == 1

        result = manager.delete_voice("delete_me")
        assert result is True
        assert len(manager.list_voices()) == 0

    def test_delete_nonexistent(self, tmp_path):
        from server.voice_manager import VoiceManager

        manager = VoiceManager(tmp_path / "voices")
        assert manager.delete_voice("nonexistent") is False

    def test_audio_normalization(self, tmp_path):
        from server.voice_manager import VoiceManager

        manager = VoiceManager(tmp_path / "voices")

        # Create stereo audio
        audio_path = tmp_path / "stereo.wav"
        stereo = np.random.randn(48000, 2).astype(np.float32) * 0.5
        sf.write(str(audio_path), stereo, 24000)

        profile = manager.create_voice(name="Stereo Test", audio_path=audio_path)

        # Verify it was converted to mono
        ref_path = manager.get_ref_audio_path(profile.voice_id)
        data, sr = sf.read(str(ref_path))
        assert data.ndim == 1  # Mono


class TestTTSModels:
    """Test TTS model base classes."""

    def test_tts_request_defaults(self):
        from server.models.base import TTSRequest

        req = TTSRequest(text="Hello")
        assert req.text == "Hello"
        assert req.speed == 1.0
        assert req.seed == -1
        assert req.voice_id is None

    def test_tts_result(self):
        from server.models.base import TTSResult

        audio = np.zeros(24000, dtype=np.float32)
        result = TTSResult(audio=audio, sample_rate=24000, duration_sec=1.0)
        assert result.duration_sec == 1.0
        assert result.sample_rate == 24000

    def test_backend_info(self):
        from server.models.base import TTSBackend, TTSRequest, TTSResult

        class DummyBackend(TTSBackend):
            name = "dummy"
            supports_cloning = True

            def load(self): pass
            def unload(self): pass
            def generate(self, req): return TTSResult(audio=np.zeros(100), sample_rate=24000)
            def is_loaded(self): return True

        backend = DummyBackend()
        info = backend.get_info()
        assert info["name"] == "dummy"
        assert info["loaded"] is True
        assert info["supports_cloning"] is True
