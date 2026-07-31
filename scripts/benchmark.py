"""
Benchmark script — compares TTS backends on speed, quality, and resource usage.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --backends f5tts,chatterbox --iterations 5
"""

import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import soundfile as sf


def _create_test_audio(path: Path, duration: float = 5.0, sr: int = 24000) -> None:
    """Create a simple test audio file if no reference is available."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Generate a simple sine wave as placeholder
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)


@click.command()
@click.option(
    "--backends", "-b",
    default="f5tts,chatterbox",
    help="Comma-separated list of backends to benchmark",
)
@click.option(
    "--ref-audio",
    default="",
    type=str,
    help="Path to reference audio for cloning test",
)
@click.option(
    "--iterations", "-n",
    default=3,
    type=int,
    help="Number of iterations per test",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cuda", "cpu"]),
)
@click.option(
    "--output-dir",
    default="benchmarks/results",
    type=click.Path(path_type=Path),
)
def main(backends: str, ref_audio: str, iterations: int, device: str, output_dir: Path):
    """Benchmark TTS backends for speed and resource usage."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]🏁 TTS Backend Benchmark[/]\n")

    backend_list = [b.strip() for b in backends.split(",") if b.strip()]

    test_texts = [
        "Hello, this is a simple test of the voice cloning system.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "In a world of artificial intelligence, the ability to generate natural-sounding "
        "speech from text has become increasingly important for many applications.",
    ]

    # Ensure we have reference audio
    ref_path = Path(ref_audio) if ref_audio else None
    if ref_path is None or not ref_path.exists():
        ref_path = Path("benchmarks/test_ref.wav")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        if not ref_path.exists():
            console.print("[yellow]No reference audio provided — creating synthetic test audio[/]")
            console.print("[yellow]For accurate cloning quality results, provide real speech audio[/]")
            _create_test_audio(ref_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for backend_name in backend_list:
        console.print(f"\n[bold]Testing backend: [cyan]{backend_name}[/][/]")

        try:
            from server.models.base import TTSRequest

            if backend_name == "f5tts":
                from server.models.f5tts_backend import F5TTSBackend
                backend = F5TTSBackend(device=device)
            elif backend_name == "chatterbox":
                from server.models.chatterbox_backend import ChatterboxBackend
                backend = ChatterboxBackend(device=device)
            elif backend_name == "qwen3tts":
                from server.models.qwen3tts_backend import Qwen3TTSBackend
                backend = Qwen3TTSBackend(device=device)
            else:
                console.print(f"[red]Unknown backend: {backend_name}[/]")
                continue

            # Load
            load_start = time.time()
            backend.load()
            load_time = time.time() - load_start
            console.print(f"  Model loaded in {load_time:.1f}s")

            # Get VRAM usage after loading
            vram_after_load = 0
            try:
                import torch
                if torch.cuda.is_available():
                    vram_after_load = torch.cuda.memory_allocated(0) / 1e9
            except Exception:
                pass

            # Run benchmark
            timings = []
            durations = []

            for i in range(iterations):
                for text in test_texts:
                    request = TTSRequest(
                        text=text,
                        ref_audio_path=ref_path,
                        ref_text="",
                    )

                    try:
                        result = backend.generate(request)
                        timings.append(result.inference_time_sec)
                        durations.append(result.duration_sec)
                    except Exception as e:
                        console.print(f"  [red]Generation failed: {e}[/]")

            backend.unload()

            if timings:
                avg_time = np.mean(timings)
                avg_duration = np.mean(durations)
                rtf = avg_time / avg_duration if avg_duration > 0 else 0

                results[backend_name] = {
                    "load_time_sec": round(load_time, 2),
                    "avg_inference_sec": round(avg_time, 3),
                    "avg_audio_duration_sec": round(avg_duration, 2),
                    "rtf": round(rtf, 3),
                    "min_inference_sec": round(min(timings), 3),
                    "max_inference_sec": round(max(timings), 3),
                    "vram_gb": round(vram_after_load, 2),
                    "iterations": len(timings),
                }

        except ImportError as e:
            console.print(f"  [red]Not installed: {e}[/]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/]")

    # Display results
    if results:
        console.print("\n")
        table = Table(title="Benchmark Results", show_lines=True)
        table.add_column("Backend", style="cyan", no_wrap=True)
        table.add_column("Load Time", justify="right")
        table.add_column("Avg Inference", justify="right")
        table.add_column("Avg Duration", justify="right")
        table.add_column("RTF", justify="right")
        table.add_column("VRAM (GB)", justify="right")

        for name, r in results.items():
            rtf_style = "green" if r["rtf"] < 1.0 else "yellow" if r["rtf"] < 2.0 else "red"
            table.add_row(
                name,
                f"{r['load_time_sec']:.1f}s",
                f"{r['avg_inference_sec']:.3f}s",
                f"{r['avg_audio_duration_sec']:.1f}s",
                f"[{rtf_style}]{r['rtf']:.2f}x[/]",
                f"{r['vram_gb']:.1f}",
            )

        console.print(table)

        # Save results
        results_file = output_dir / "benchmark_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        console.print(f"\n[dim]Results saved to {results_file}[/]")
    else:
        console.print("[red]No results to display[/]")


if __name__ == "__main__":
    main()
