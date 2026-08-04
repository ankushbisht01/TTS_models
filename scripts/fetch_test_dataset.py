"""
Fetch a public-domain single-speaker corpus for testing the fine-tuning pipeline.

Downloads audio into a directory that scripts/prepare_dataset.py can consume,
so the whole path — segmentation, transcription, Arrow conversion, training —
can be exercised on data that is free to use.

LJSpeech-1.1 is public domain: LibriVox recordings of Project Gutenberg texts,
single female speaker, ~24 hours total.

The archive is streamed and extraction stops once --limit clips are written, so
a smoke test pulls a few MB rather than the full 2.7 GB.

Usage:
    python -m scripts.fetch_test_dataset --limit 50
    python -m scripts.prepare_dataset -i ./raw_audio -o ./training_data \
        -s ljspeech --transcribe
"""

import tarfile
import urllib.request
from pathlib import Path

import click

DATASETS = {
    "ljspeech": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "mode": "r|bz2",
        "wav_dir": "wavs/",
        "size_gb": 2.7,
        "speakers": 1,
        "license": "Public domain (LibriVox / Project Gutenberg)",
    },
    "libritts": {
        "url": "https://www.openslr.org/resources/60/dev-clean.tar.gz",
        "mode": "r|gz",
        "wav_dir": "",  # nested per speaker/chapter
        "size_gb": 1.3,
        "speakers": 40,
        "license": "CC BY 4.0",
    },
}


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(sorted(DATASETS)),
    default="ljspeech",
    help="Corpus to fetch (ljspeech is single-speaker, best for fine-tuning tests)",
)
@click.option(
    "--output-dir", "-o",
    default=Path("./raw_audio"),
    type=click.Path(path_type=Path),
    help="Where to write the audio files",
)
@click.option(
    "--limit",
    default=50,
    type=int,
    help="Stop after this many clips (0 = download everything)",
)
def main(dataset: str, output_dir: Path, limit: int):
    """Download a public-domain corpus for pipeline testing."""
    from rich.console import Console

    console = Console()
    spec = DATASETS[dataset]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]📥 Fetching {dataset}[/]\n")
    console.print(f"  License  : {spec['license']}")
    console.print(f"  Speakers : {spec['speakers']}")
    console.print(f"  Full size: ~{spec['size_gb']} GB")
    console.print(f"  Limit    : {limit or 'none — full download'}")
    console.print(f"  Output   : {output_dir}\n")

    if limit == 0 and not click.confirm(
        f"This downloads ~{spec['size_gb']} GB. Continue?", default=False
    ):
        console.print("[yellow]Aborted.[/]")
        return

    count = 0
    bytes_read = 0

    # Stream the archive so --limit can stop early instead of pulling
    # gigabytes to keep a handful of clips.
    req = urllib.request.Request(
        spec["url"], headers={"User-Agent": "voice-cloning-server/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode=spec["mode"]) as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                if spec["wav_dir"] and spec["wav_dir"] not in member.name:
                    continue

                src = tar.extractfile(member)
                if src is None:
                    continue

                data = src.read()
                dest = output_dir / Path(member.name).name
                dest.write_bytes(data)

                count += 1
                bytes_read += len(data)

                if count % 10 == 0:
                    console.print(
                        f"  {count} clips ({bytes_read / 1e6:.1f} MB)", end="\r"
                    )

                if limit and count >= limit:
                    break

    console.print(f"\n[green]✅ Downloaded {count} clips[/] "
                  f"({bytes_read / 1e6:.1f} MB) to {output_dir}")

    if count == 0:
        raise click.ClickException(
            "No audio extracted — the archive layout may have changed."
        )

    console.print(
        f"\nNext:\n"
        f"  [cyan]python -m scripts.prepare_dataset "
        f"-i {output_dir} -o ./training_data -s {dataset} --transcribe[/]"
    )


if __name__ == "__main__":
    main()
