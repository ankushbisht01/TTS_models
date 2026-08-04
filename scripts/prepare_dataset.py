"""
Dataset preparation for F5-TTS fine-tuning.

Takes a directory of audio files, segments and normalizes them, optionally
transcribes them with Whisper, and writes the pipe-delimited CSV that
`f5-tts_finetune-cli` ingests.

Output layout:
    <output-dir>/wavs/<speaker>_00000.wav
    <output-dir>/metadata.csv          audio_file|text   (absolute paths)
    <output-dir>/dataset_info.json

Usage:
    python -m scripts.prepare_dataset \
        --input-dir ./raw_audio/ \
        --output-dir ./training_data/ \
        --speaker-name "John" \
        --transcribe

Only use audio you have the rights to, and for a real person's voice, their
consent. Then run scripts/finetune_f5tts.sh to train.
"""

import csv
import json
import os
from pathlib import Path

import click
import numpy as np
import soundfile as sf


def _split_on_silence(
    audio: np.ndarray,
    sr: int,
    min_duration: float,
    max_duration: float,
    threshold: float = 0.01,
) -> list[np.ndarray]:
    """
    Split audio into speech segments at silent gaps.

    Splitting mid-word produces training pairs whose transcript doesn't match
    the audio, so prefer silence boundaries and only hard-cut a run of speech
    that exceeds max_duration.
    """
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)  # 10ms hop

    if len(audio) < frame_length:
        return []

    voiced = []
    for i in range(0, len(audio) - frame_length, hop_length):
        rms = np.sqrt(np.mean(audio[i : i + frame_length] ** 2))
        voiced.append(rms > threshold)

    segments: list[np.ndarray] = []
    min_silence_frames = int(0.3 / 0.010)  # 300ms of silence ends a segment

    def emit(s_frame: int, e_frame: int) -> None:
        start = max(0, int(s_frame * hop_length))
        end = min(int(e_frame * hop_length) + frame_length, len(audio))
        chunk = audio[start:end]
        if len(chunk) / sr < min_duration:
            return
        max_samples = int(max_duration * sr)
        for off in range(0, len(chunk), max_samples):
            piece = chunk[off : off + max_samples]
            if len(piece) / sr >= min_duration:
                segments.append(piece)

    start_frame = None
    silence_run = 0

    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if start_frame is None:
                start_frame = i
            silence_run = 0
        elif start_frame is not None:
            silence_run += 1
            if silence_run >= min_silence_frames:
                emit(start_frame, i - silence_run)
                start_frame = None
                silence_run = 0

    if start_frame is not None:
        emit(start_frame, len(voiced) - 1)

    return segments


def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader — avoids a python-dotenv dependency."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def _groq_model_name(model_size: str) -> str:
    """Map local Whisper names to Groq's model ids."""
    if model_size.startswith(("whisper-", "distil-")):
        return model_size
    return f"whisper-{model_size}"


def _load_groq_transcriber(model_size: str, api_key: str):
    """Transcribe via Groq's OpenAI-compatible endpoint."""
    try:
        from groq import Groq
    except ImportError:
        raise click.ClickException(
            "The groq package is required for --transcribe-backend groq:\n"
            "  .venv/bin/pip install groq"
        )

    import time as _time

    client = Groq(api_key=api_key)
    model = _groq_model_name(model_size)

    def transcribe(path: Path) -> str:
        last_err = None
        for attempt in range(5):
            try:
                with open(path, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        file=(path.name, f.read()),
                        model=model,
                        response_format="text",
                    )
                return (resp if isinstance(resp, str) else resp.text).strip()
            except Exception as e:  # rate limits are the common case
                last_err = e
                if "rate" in str(e).lower() or "429" in str(e):
                    _time.sleep(2**attempt)
                    continue
                raise
        raise click.ClickException(f"Groq transcription failed for {path.name}: {last_err}")

    return transcribe


def _load_transcriber(model_size: str, device: str):
    """Load faster-whisper, falling back to openai-whisper."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device.startswith("cuda") else "int8",
        )

        def transcribe(path: Path) -> str:
            segs, _ = model.transcribe(str(path), beam_size=5)
            return " ".join(s.text.strip() for s in segs).strip()

        return transcribe
    except ImportError:
        pass

    try:
        import whisper

        model = whisper.load_model(model_size, device=device)

        def transcribe(path: Path) -> str:
            return model.transcribe(str(path))["text"].strip()

        return transcribe
    except ImportError:
        raise click.ClickException(
            "--transcribe needs a Whisper implementation. Install one INTO THE VENV\n"
            "(plain `pip install` hits PEP 668 'externally-managed-environment'\n"
            "on Arch/Debian):\n"
            "  .venv/bin/pip install faster-whisper   (recommended, GPU)\n"
            "  .venv/bin/pip install openai-whisper"
        )


@click.command()
@click.option(
    "--input-dir", "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory of source audio files",
)
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for prepared dataset",
)
@click.option(
    "--speaker-name", "-s",
    default="speaker",
    help="Name of the speaker",
)
@click.option(
    "--target-sr",
    default=24000,
    type=int,
    help="Target sample rate",
)
@click.option(
    "--max-duration",
    default=15.0,
    type=float,
    help="Maximum duration per segment (seconds)",
)
@click.option(
    "--min-duration",
    default=2.0,
    type=float,
    help="Minimum duration per segment (seconds)",
)
@click.option(
    "--transcribe",
    is_flag=True,
    help="Auto-transcribe segments with Whisper (fills the text column)",
)
@click.option(
    "--transcribe-backend",
    type=click.Choice(["auto", "groq", "faster-whisper", "whisper"]),
    default="auto",
    help="auto = Groq if GROQ_API_KEY is set, else local Whisper",
)
@click.option(
    "--whisper-model",
    default="large-v3",
    help="Whisper model size ('large-v3'; Groq maps it to 'whisper-large-v3')",
)
@click.option(
    "--device",
    default="cuda",
    help="Device for local Whisper",
)
def main(
    input_dir: Path,
    output_dir: Path,
    speaker_name: str,
    target_sr: int,
    max_duration: float,
    min_duration: float,
    transcribe: bool,
    transcribe_backend: str,
    whisper_model: str,
    device: str,
):
    """Prepare an audio dataset for F5-TTS fine-tuning."""
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    console.print("[bold cyan]📦 Dataset Preparation[/]\n")

    output_dir = output_dir.resolve()
    audio_dir = output_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    audio_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in audio_extensions
    )

    if not audio_files:
        raise click.ClickException(f"No audio files found in {input_dir}")

    console.print(f"Found [cyan]{len(audio_files)}[/] audio files")

    transcriber = None
    if transcribe:
        # GROQ_API_KEY from the real environment first, then the project .env
        api_key = os.environ.get("GROQ_API_KEY") or _read_dotenv(
            Path(__file__).resolve().parent.parent / ".env"
        ).get("GROQ_API_KEY", "")

        backend = transcribe_backend
        if backend == "auto":
            backend = "groq" if api_key else "faster-whisper"

        if backend == "groq":
            if not api_key:
                raise click.ClickException(
                    "GROQ_API_KEY is not set. Add it to .env:\n"
                    "  echo 'GROQ_API_KEY=gsk_...' >> .env\n"
                    "or export it, or use --transcribe-backend faster-whisper."
                )
            console.print(
                f"Transcribing via [cyan]Groq {_groq_model_name(whisper_model)}[/]"
            )
            console.print(
                "[yellow]Note: this uploads each audio segment to Groq's API.[/]"
            )
            transcriber = _load_groq_transcriber(whisper_model, api_key)
        else:
            console.print(f"Loading Whisper [cyan]{whisper_model}[/] on {device}...")
            transcriber = _load_transcriber(whisper_model, device)

    metadata: list[dict] = []
    segment_count = 0

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(audio_files))

        for audio_file in audio_files:
            try:
                data, sr = sf.read(str(audio_file))

                if data.ndim > 1:
                    data = data.mean(axis=1)

                if sr != target_sr:
                    import librosa

                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

                peak = np.abs(data).max()
                if peak > 0:
                    data = data / peak * 0.95

                for chunk in _split_on_silence(data, sr, min_duration, max_duration):
                    segment_name = f"{speaker_name}_{segment_count:05d}.wav"
                    segment_path = audio_dir / segment_name
                    sf.write(str(segment_path), chunk, sr)

                    text = transcriber(segment_path) if transcriber else ""

                    metadata.append(
                        {
                            # f5-tts requires ABSOLUTE paths in the CSV
                            "audio_file": str(segment_path),
                            "text": text,
                            "duration": round(len(chunk) / sr, 2),
                        }
                    )
                    segment_count += 1

            except Exception as e:
                console.print(f"[red]Error processing {audio_file.name}: {e}[/]")

            progress.advance(task)

    if not metadata:
        raise click.ClickException(
            "No usable segments produced. Try lowering --min-duration."
        )

    # f5-tts's prepare_csv_wavs expects: header, "|" delimiter, two columns.
    # A "," delimiter or extra columns silently mis-parses transcripts.
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["audio_file", "text"])
        for m in metadata:
            writer.writerow([m["audio_file"], m["text"]])

    total_duration = sum(m["duration"] for m in metadata)
    untranscribed = sum(1 for m in metadata if not m["text"].strip())

    summary = {
        "speaker": speaker_name,
        "total_segments": segment_count,
        "total_duration_sec": round(total_duration, 1),
        "total_duration_min": round(total_duration / 60, 1),
        "sample_rate": target_sr,
        "source_files": len(audio_files),
        "untranscribed_segments": untranscribed,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(summary, indent=2))

    console.print("\n[green]✅ Dataset prepared![/]")
    console.print(f"  Segments : [cyan]{segment_count}[/]")
    console.print(f"  Duration : [cyan]{total_duration / 60:.1f} min[/]")
    console.print(f"  Metadata : {meta_path}")

    if total_duration < 10 * 60:
        console.print(
            f"\n[yellow]⚠️  Only {total_duration / 60:.1f} min of audio. "
            "Fine-tuning F5-TTS wants 30+ min; below that, zero-shot cloning "
            "via /v1/voices/create usually gives better results.[/]"
        )

    if untranscribed:
        console.print(
            f"\n[yellow]⚠️  {untranscribed} segments have no transcript. "
            "Fill the text column in metadata.csv or re-run with --transcribe — "
            "F5-TTS trains on (audio, text) pairs and empty text corrupts training.[/]"
        )
    else:
        console.print(
            f"\nNext: [cyan]bash scripts/finetune_f5tts.sh {output_dir} {speaker_name}[/]"
        )


if __name__ == "__main__":
    main()
