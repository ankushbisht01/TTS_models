"""
Dataset preparation for LoRA fine-tuning.

Takes a directory of audio files and creates a properly formatted dataset
for fine-tuning F5-TTS with LoRA adapters.

Usage:
    python -m scripts.prepare_dataset \
        --input-dir ./raw_audio/ \
        --output-dir ./training_data/ \
        --speaker-name "John"
"""

import csv
import json
from pathlib import Path

import click
import numpy as np
import soundfile as sf


def _detect_silence(audio: np.ndarray, sr: int, threshold: float = 0.01) -> list[tuple[float, float]]:
    """Detect silent segments for splitting."""
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)  # 10ms hop

    segments = []
    start = None

    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i : i + frame_length]
        rms = np.sqrt(np.mean(frame**2))

        if rms > threshold and start is None:
            start = i / sr
        elif rms <= threshold and start is not None:
            end = i / sr
            if end - start >= 1.0:  # Minimum 1 second segments
                segments.append((start, end))
            start = None

    if start is not None:
        end = len(audio) / sr
        if end - start >= 1.0:
            segments.append((start, end))

    return segments


@click.command()
@click.option(
    "--input-dir", "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing raw audio files",
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
def main(
    input_dir: Path,
    output_dir: Path,
    speaker_name: str,
    target_sr: int,
    max_duration: float,
    min_duration: float,
):
    """Prepare audio dataset for voice cloning fine-tuning."""
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    console.print(f"[bold cyan]📦 Dataset Preparation[/]\n")

    audio_dir = output_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ])

    if not audio_files:
        console.print(f"[red]No audio files found in {input_dir}[/]")
        return

    console.print(f"Found [cyan]{len(audio_files)}[/] audio files")

    metadata = []
    segment_count = 0

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(audio_files))

        for audio_file in audio_files:
            try:
                # Load audio
                data, sr = sf.read(str(audio_file))

                # Convert to mono
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # Resample if needed
                if sr != target_sr:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

                # Normalize
                peak = np.abs(data).max()
                if peak > 0:
                    data = data / peak * 0.95

                # Split into segments if too long
                total_duration = len(data) / sr

                if total_duration <= max_duration:
                    # Save as single file
                    segment_name = f"{speaker_name}_{segment_count:05d}.wav"
                    sf.write(str(audio_dir / segment_name), data, sr)
                    metadata.append({
                        "audio_file": f"wavs/{segment_name}",
                        "text": "",  # User needs to fill in transcripts
                        "speaker": speaker_name,
                        "duration": round(total_duration, 2),
                    })
                    segment_count += 1
                else:
                    # Split into chunks
                    samples_per_chunk = int(max_duration * sr)
                    for start_sample in range(0, len(data), samples_per_chunk):
                        chunk = data[start_sample : start_sample + samples_per_chunk]
                        chunk_duration = len(chunk) / sr

                        if chunk_duration >= min_duration:
                            segment_name = f"{speaker_name}_{segment_count:05d}.wav"
                            sf.write(str(audio_dir / segment_name), chunk, sr)
                            metadata.append({
                                "audio_file": f"wavs/{segment_name}",
                                "text": "",
                                "speaker": speaker_name,
                                "duration": round(chunk_duration, 2),
                            })
                            segment_count += 1

            except Exception as e:
                console.print(f"[red]Error processing {audio_file.name}: {e}[/]")

            progress.advance(task)

    # Write metadata
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file", "text", "speaker", "duration"])
        writer.writeheader()
        writer.writerows(metadata)

    # Write summary
    summary = {
        "speaker": speaker_name,
        "total_segments": segment_count,
        "total_duration_sec": sum(m["duration"] for m in metadata),
        "sample_rate": target_sr,
        "source_files": len(audio_files),
    }

    summary_path = output_dir / "dataset_info.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    console.print(f"\n[green]✅ Dataset prepared![/]")
    console.print(f"  Segments: [cyan]{segment_count}[/]")
    console.print(f"  Total duration: [cyan]{summary['total_duration_sec']:.1f}s[/]")
    console.print(f"  Output: [cyan]{output_dir}[/]")
    console.print(f"\n[yellow]⚠️  Remember to fill in transcripts in metadata.csv![/]")


if __name__ == "__main__":
    main()
