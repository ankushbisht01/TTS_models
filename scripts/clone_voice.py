"""
CLI tool for quick voice cloning from the command line.

Usage:
    python -m scripts.clone_voice \
        --ref speaker.wav \
        --text "Hello world" \
        --output output.wav \
        --backend f5tts
"""

import sys
import time
from pathlib import Path

import click
import numpy as np
import soundfile as sf


@click.command()
@click.option(
    "--ref", "-r",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to reference audio file (5-30 seconds WAV/MP3/FLAC)",
)
@click.option(
    "--text", "-t",
    required=True,
    help="Text to synthesize in the cloned voice",
)
@click.option(
    "--ref-text",
    default="",
    help="Transcript of the reference audio (improves quality for F5-TTS)",
)
@click.option(
    "--output", "-o",
    default="output.wav",
    type=click.Path(path_type=Path),
    help="Output audio file path",
)
@click.option(
    "--backend", "-b",
    default="f5tts",
    type=click.Choice(["f5tts", "chatterbox", "qwen3tts"]),
    help="TTS backend to use",
)
@click.option("--speed", default=1.0, type=float, help="Speaking speed (0.25-4.0)")
@click.option("--seed", default=-1, type=int, help="Random seed (-1 = random)")
@click.option(
    "--exaggeration",
    default=0.5,
    type=float,
    help="Emotion exaggeration (0.0-1.0, Chatterbox only)",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to run inference on",
)
def main(
    ref: Path,
    text: str,
    ref_text: str,
    output: Path,
    backend: str,
    speed: float,
    seed: int,
    exaggeration: float,
    device: str,
):
    """Clone a voice and generate speech from text."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        f"[bold cyan]Voice Cloning CLI[/]\n"
        f"Backend: [yellow]{backend}[/]\n"
        f"Reference: [green]{ref.name}[/]\n"
        f"Text: [white]{text[:80]}{'...' if len(text) > 80 else ''}[/]",
        title="🎤 Voice Cloner",
    ))

    # Import and load the backend
    console.print("[dim]Loading model...[/]")
    load_start = time.time()

    from server.models.base import TTSRequest

    if backend == "f5tts":
        from server.models.f5tts_backend import F5TTSBackend
        model = F5TTSBackend(device=device)
    elif backend == "chatterbox":
        from server.models.chatterbox_backend import ChatterboxBackend
        model = ChatterboxBackend(device=device, default_exaggeration=exaggeration)
    elif backend == "qwen3tts":
        from server.models.qwen3tts_backend import Qwen3TTSBackend
        model = Qwen3TTSBackend(device=device)
    else:
        console.print(f"[red]Unknown backend: {backend}[/]")
        sys.exit(1)

    model.load()
    load_time = time.time() - load_start
    console.print(f"[green]✅ Model loaded in {load_time:.1f}s[/]")

    # Generate
    console.print("[dim]Generating speech...[/]")

    request = TTSRequest(
        text=text,
        ref_audio_path=ref,
        ref_text=ref_text,
        speed=speed,
        seed=seed,
        exaggeration=exaggeration,
    )

    result = model.generate(request)

    # Save output
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), result.audio, result.sample_rate)

    # Summary
    rtf = result.inference_time_sec / result.duration_sec if result.duration_sec > 0 else 0

    console.print(Panel.fit(
        f"[green]✅ Done![/]\n"
        f"Output: [cyan]{output}[/]\n"
        f"Duration: [yellow]{result.duration_sec:.2f}s[/]\n"
        f"Inference: [yellow]{result.inference_time_sec:.2f}s[/]\n"
        f"RTF: [yellow]{rtf:.2f}x[/] {'(faster than real-time)' if rtf < 1 else ''}",
        title="Results",
    ))

    model.unload()


if __name__ == "__main__":
    main()
