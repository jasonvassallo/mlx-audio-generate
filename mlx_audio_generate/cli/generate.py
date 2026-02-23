import argparse


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

    if args.model == "stable_audio":
        from mlx_audio_generate.models.stable_audio import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(args.weights_dir)
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

        pipe = MusicGenPipeline.from_pretrained(args.weights_dir)
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

    output_path = args.output or f"{args.model}_output.wav"

    from mlx_audio_generate.shared.audio_io import save_wav

    save_wav(output_path, audio, sample_rate=sample_rate, channels=channels)
    print(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
