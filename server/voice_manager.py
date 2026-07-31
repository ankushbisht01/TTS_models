"""
Voice profile manager — handles storing, loading, and listing voice profiles.

A voice profile is a directory containing:
  - ref_audio.wav    — the reference audio clip
  - metadata.json    — name, description, transcript, backend preferences
"""

import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """A saved voice profile for consistent cloning."""

    voice_id: str
    name: str
    description: str = ""
    ref_text: str = ""  # Transcript of the reference audio
    ref_audio_file: str = "ref_audio.wav"
    sample_rate: int = 24000
    duration_sec: float = 0.0
    preferred_backend: str = ""  # Which backend works best for this voice
    created_at: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        # Handle extra keys gracefully
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class VoiceManager:
    """Manages voice profiles on disk."""

    def __init__(self, voices_dir: Path):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, VoiceProfile] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan voices directory and cache all profiles."""
        self._cache.clear()
        for meta_file in self.voices_dir.rglob("metadata.json"):
            try:
                data = json.loads(meta_file.read_text())
                profile = VoiceProfile.from_dict(data)
                self._cache[profile.voice_id] = profile
            except Exception as e:
                logger.warning("Failed to load voice profile from %s: %s", meta_file, e)

        logger.info("Loaded %d voice profiles", len(self._cache))

    def list_voices(self) -> list[VoiceProfile]:
        """List all available voice profiles."""
        return list(self._cache.values())

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Get a voice profile by ID."""
        return self._cache.get(voice_id)

    def get_ref_audio_path(self, voice_id: str) -> Path | None:
        """Get the reference audio file path for a voice."""
        profile = self.get_voice(voice_id)
        if profile is None:
            return None
        audio_path = self.voices_dir / voice_id / profile.ref_audio_file
        return audio_path if audio_path.exists() else None

    def create_voice(
        self,
        name: str,
        audio_path: Path | str,
        ref_text: str = "",
        description: str = "",
        preferred_backend: str = "",
        tags: list[str] | None = None,
    ) -> VoiceProfile:
        """
        Create a new voice profile from an audio file.

        The audio file is copied into the voices directory.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Generate a stable ID from name (or use UUID)
        voice_id = name.lower().replace(" ", "_").replace("-", "_")
        if self.get_voice(voice_id):
            voice_id = f"{voice_id}_{uuid.uuid4().hex[:6]}"

        # Create profile directory
        profile_dir = self.voices_dir / voice_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Copy and normalize audio
        dest_audio = profile_dir / "ref_audio.wav"
        self._normalize_audio(audio_path, dest_audio)

        # Get audio info
        info = sf.info(str(dest_audio))

        from datetime import datetime, timezone

        profile = VoiceProfile(
            voice_id=voice_id,
            name=name,
            description=description,
            ref_text=ref_text,
            ref_audio_file="ref_audio.wav",
            sample_rate=info.samplerate,
            duration_sec=info.duration,
            preferred_backend=preferred_backend,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
        )

        # Save metadata
        meta_path = profile_dir / "metadata.json"
        meta_path.write_text(json.dumps(profile.to_dict(), indent=2))

        self._cache[voice_id] = profile
        logger.info("Created voice profile: %s (%s)", name, voice_id)

        return profile

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice profile."""
        profile = self.get_voice(voice_id)
        if profile is None:
            return False

        profile_dir = self.voices_dir / voice_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

        self._cache.pop(voice_id, None)
        logger.info("Deleted voice profile: %s", voice_id)
        return True

    def _normalize_audio(self, src: Path, dest: Path) -> None:
        """
        Copy and normalize audio to a consistent format.
        Converts to WAV, mono, and trims to max_duration if needed.
        """
        import numpy as np

        data, sr = sf.read(str(src))

        # Convert to mono if stereo
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Normalize amplitude
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.95

        # Trim to 30 seconds max
        max_samples = int(30 * sr)
        if len(data) > max_samples:
            data = data[:max_samples]
            logger.info("Trimmed reference audio to 30 seconds")

        sf.write(str(dest), data, sr)
