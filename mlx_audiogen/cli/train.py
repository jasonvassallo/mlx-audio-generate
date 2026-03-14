"""CLI entry point for LoRA training: mlx-audiogen-train.

Usage:
    mlx-audiogen-train \\
        --data ./my-music/ --base-model musicgen-small \\
        --name my-style
    mlx-audiogen-train \\
        --data ./my-music/ --base-model musicgen-small \\
        --name my-style --profile deep
    mlx-audiogen-train \\
        --data ./my-music/ --base-model musicgen-small \\
        --name my-style --rank 32 --targets q_proj,v_proj
"""

import argparse
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LoRA style adapter for MusicGen"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data directory (WAV files + optional metadata.jsonl)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name (e.g., musicgen-small) or path to weights directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the LoRA adapter (alphanumeric, hyphens, underscores)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "deep"],
        help="Training profile preset (default: balanced)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/.mlx-audiogen/loras/<name>/)",
    )
    # Advanced overrides
    parser.add_argument(
        "--rank", type=int, default=None, help="LoRA rank (overrides profile)"
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="LoRA alpha (overrides profile)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target layers (e.g., q_proj,v_proj,out_proj). "
        "Prefix with encoder_attn. for cross-attention targets.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Audio chunk duration in seconds (5-40, default 10)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Validate inputs
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", args.name):
        print("Error: --name must be 1-64 chars, alphanumeric/hyphens/underscores only")
        sys.exit(1)

    data_dir = Path(args.data).resolve()
    if ".." in Path(args.data).parts:
        print("Error: --data path must not contain '..'")
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)

    if args.chunk_seconds < 5 or args.chunk_seconds > 40:
        print("Error: --chunk-seconds must be between 5 and 40")
        sys.exit(1)

    if args.epochs < 1 or args.epochs > 100:
        print("Error: --epochs must be between 1 and 100")
        sys.exit(1)

    if args.seed is not None:
        mx.random.seed(args.seed)
        np.random.seed(args.seed)

    # Build config from profile + overrides
    from mlx_audiogen.lora.config import PROFILES, LoRAConfig

    profile = PROFILES[args.profile]
    rank = args.rank if args.rank is not None else profile.rank
    alpha = args.alpha if args.alpha is not None else profile.alpha

    if args.targets:
        # Parse targets: bare names default to self_attn
        raw_targets = [t.strip() for t in args.targets.split(",")]
        targets = []
        for t in raw_targets:
            if "." in t:
                targets.append(t)  # Already qualified
            else:
                targets.append(f"self_attn.{t}")
    else:
        targets = profile.targets

    # Load pipeline
    print(f"Loading base model: {args.base_model}")
    from mlx_audiogen.models.musicgen.pipeline import MusicGenPipeline

    pipeline = MusicGenPipeline.from_pretrained(weights_dir=args.base_model)
    hidden_size = pipeline.config.decoder.hidden_size

    config = LoRAConfig(
        name=args.name,
        base_model=args.base_model,
        hidden_size=hidden_size,
        rank=rank,
        alpha=alpha,
        targets=targets,
        profile=args.profile,
        chunk_seconds=args.chunk_seconds,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        early_stop=not args.no_early_stop,
        patience=args.patience,
    )

    # Load and prepare dataset
    print(f"Loading training data from: {data_dir}")
    from mlx_audiogen.lora.dataset import (
        apply_delay_pattern,
        chunk_audio,
        load_and_prepare_audio,
        scan_dataset,
    )

    entries = scan_dataset(data_dir)
    print(f"Found {len(entries)} audio files")

    # Encode audio chunks through EnCodec
    training_data = []
    for entry in entries:
        audio = load_and_prepare_audio(entry["file"], target_sr=pipeline.sample_rate)
        chunks = chunk_audio(audio, pipeline.sample_rate, args.chunk_seconds)

        for chunk in chunks:
            # EnCodec encode expects (batch, time, channels)
            chunk_mx = mx.array(chunk)[mx.newaxis, :, mx.newaxis]  # (1, T, 1)
            codes, _scales = pipeline.encodec.encode(chunk_mx)
            # codes shape from RVQ: (B, K, T) -> transpose to (B, T, K)
            tokens = mx.swapaxes(codes, 1, 2)

            # Apply delay pattern
            delayed, valid = apply_delay_pattern(
                tokens,
                pipeline.config.decoder.num_codebooks,
                pipeline.config.decoder.bos_token_id,
            )

            training_data.append(
                {
                    "delayed_tokens": delayed,
                    "valid_mask": valid,
                    "text": entry["text"],
                }
            )

    print(
        f"Prepared {len(training_data)} training samples ({args.chunk_seconds}s chunks)"
    )

    if not training_data:
        print("Error: No valid training samples produced")
        sys.exit(1)

    # Train
    from mlx_audiogen.lora.trainer import LoRATrainer

    output_dir = Path(args.output) if args.output else None
    trainer = LoRATrainer(
        pipeline=pipeline,
        config=config,
        training_data=training_data,
        output_dir=output_dir,
    )
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
