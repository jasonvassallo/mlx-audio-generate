import argparse
import sys
from pathlib import Path


def _validate_output_path(path_str: str) -> Path:
    """Validate output path is safe (no traversal, writable location)."""
    path = Path(path_str).resolve()
    # Reject paths that try to escape via '..'
    if ".." in Path(path_str).parts:
        print(f"Error: Output path must not contain '..': {path_str}")
        sys.exit(1)
    # Ensure parent directory exists or can be created
    parent = path.parent
    if not parent.exists():
        print(f"Error: Output directory does not exist: {parent}")
        sys.exit(1)
    return path


def _validate_weights_dir(path_str: str) -> Path:
    """Validate weights directory exists and contains expected files."""
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"Error: Weights directory does not exist: {path_str}")
        sys.exit(1)
    if not path.is_dir():
        print(f"Error: Weights path is not a directory: {path_str}")
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser(description="MLX Audio Generate")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["musicgen", "stable_audio"],
        help="Model to use for generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing desired audio",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (stable_audio only)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    # MusicGen-specific
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument(
        "--guidance-coef",
        type=float,
        default=3.0,
        help="Classifier-free guidance coefficient (musicgen only)",
    )
    # Stable Audio-specific
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Diffusion steps (stable_audio only)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=6.0,
        help="CFG scale (stable_audio only)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="ODE sampler (stable_audio only)",
    )
    # General
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Path to converted weights directory",
    )

    args = parser.parse_args()

    # --- Validate inputs ---
    # C3: Validate weights directory
    if args.weights_dir is not None:
        weights_dir = str(_validate_weights_dir(args.weights_dir))
    else:
        weights_dir = None

    # M2: Validate numeric ranges
    if args.seconds <= 0 or args.seconds > 300:
        print("Error: --seconds must be between 0 and 300")
        sys.exit(1)
    if args.temperature <= 0:
        print("Error: --temperature must be positive")
        sys.exit(1)
    if args.top_k < 1:
        print("Error: --top-k must be at least 1")
        sys.exit(1)
    if args.steps < 1 or args.steps > 1000:
        print("Error: --steps must be between 1 and 1000")
        sys.exit(1)
    if args.cfg_scale < 0:
        print("Error: --cfg-scale must be non-negative")
        sys.exit(1)
    if args.guidance_coef < 0:
        print("Error: --guidance-coef must be non-negative")
        sys.exit(1)

    if args.model == "stable_audio":
        from mlx_audio_generate.models.stable_audio import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(weights_dir)
        audio = pipe.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seconds_total=args.seconds,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            sampler=args.sampler,
        )
        sample_rate = 44100
        channels = 2
    elif args.model == "musicgen":
        from mlx_audio_generate.models.musicgen import MusicGenPipeline

        pipe = MusicGenPipeline.from_pretrained(weights_dir)
        audio = pipe.generate(
            prompt=args.prompt,
            seconds=args.seconds,
            temperature=args.temperature,
            top_k=args.top_k,
            guidance_coef=args.guidance_coef,
            seed=args.seed,
        )
        sample_rate = pipe.sample_rate
        channels = 1

    # C1: Validate output path
    output_str = args.output or f"{args.model}_output.wav"
    output_path = _validate_output_path(output_str)

    from mlx_audio_generate.shared.audio_io import save_wav

    save_wav(str(output_path), audio, sample_rate=sample_rate, channels=channels)
    print(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
