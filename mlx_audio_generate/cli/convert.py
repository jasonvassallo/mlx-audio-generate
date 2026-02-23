import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to MLX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., facebook/musicgen-small)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlx_model",
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Data type for converted weights",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect model type from repo ID
    repo_lower = args.model.lower()
    if "musicgen" in repo_lower:
        from mlx_audio_generate.models.musicgen.convert import convert_musicgen

        convert_musicgen(args.model, output_dir, dtype=args.dtype)
    elif "stable-audio" in repo_lower or "stable_audio" in repo_lower:
        from mlx_audio_generate.models.stable_audio.convert import (
            convert_stable_audio,
        )

        convert_stable_audio(args.model, output_dir, dtype=args.dtype)
    else:
        print(f"Could not auto-detect model type from repo ID: {args.model}")
        print("Supported patterns: 'musicgen', 'stable-audio'")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
