"""Convert MusicGen weights from HuggingFace safetensors to MLX format.

Reads the HF model.safetensors and splits it into:
    - decoder.safetensors  (transformer + embeddings + LM heads + enc_to_dec_proj)
    - t5.safetensors       (T5 text encoder)
    - config.json          (model configuration)
    - t5_config.json       (T5 configuration)
    + tokenizer files

Weight key remapping:
    Decoder: Strip 'decoder.model.decoder.' prefix, rename encoder_attn -> encoder_attn
    T5: Strip 'text_encoder.' prefix, map HF T5 layout to our T5 module layout
    Audio encoder: SKIPPED (we load EnCodec separately from mlx-community)
"""

import json
from pathlib import Path

import numpy as np

from mlx_audio_generate.shared.hub import (
    download_model,
    load_pytorch_bin,
    load_safetensors,
    save_safetensors,
)


def _remap_t5_key(k: str) -> str:
    """Remap a single T5 key from HF format to our T5 module format.

    HF format:
        encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
        encoder.block.{i}.layer.0.layer_norm.weight
        encoder.block.{i}.layer.1.DenseReluDense.{wi,wo}.weight
        encoder.block.{i}.layer.1.layer_norm.weight
        encoder.final_layer_norm.weight
        shared.weight

    Our T5 module format:
        encoder.block.{i}.self_attn.{q,k,v,o}.weight
        encoder.block.{i}.self_attn_norm.weight
        encoder.block.{i}.ff.{wi,wo}.weight
        encoder.block.{i}.ff_norm.weight
        encoder.final_layer_norm.weight
        shared.weight

    Also handles relative_attention_bias at block 0.
    """
    # Self-attention
    k = k.replace(".layer.0.SelfAttention.", ".self_attn.")
    # Self-attention norm
    k = k.replace(".layer.0.layer_norm.", ".self_attn_norm.")
    # Feed-forward
    k = k.replace(".layer.1.DenseReluDense.", ".ff.")
    # Feed-forward norm
    k = k.replace(".layer.1.layer_norm.", ".ff_norm.")
    # Relative attention bias embedding
    k = k.replace(
        ".self_attn.relative_attention_bias.",
        ".self_attn.relative_attention_bias.",
    )
    return k


def _remap_decoder_key(k: str) -> str:
    """Remap a decoder key from HF format to our module format.

    HF format:
        decoder.model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight
        decoder.model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj.weight
        decoder.model.decoder.layers.{i}.self_attn_layer_norm.weight/bias
        decoder.model.decoder.layers.{i}.encoder_attn_layer_norm.weight/bias
        decoder.model.decoder.layers.{i}.fc1.weight/bias
        decoder.model.decoder.layers.{i}.fc2.weight/bias
        decoder.model.decoder.layers.{i}.final_layer_norm.weight/bias
        decoder.model.decoder.layer_norm.weight/bias
        decoder.model.decoder.embed_tokens.{k}.weight
        decoder.lm_heads.{k}.weight
        enc_to_dec_proj.weight/bias

    Our module format:
        layers.{i}.self_attn.{q,k,v,out}_proj.weight
        layers.{i}.encoder_attn.{q,k,v,out}_proj.weight
        layers.{i}.self_attn_layer_norm.weight/bias
        layers.{i}.encoder_attn_layer_norm.weight/bias
        layers.{i}.fc1.weight/bias
        layers.{i}.fc2.weight/bias
        layers.{i}.final_layer_norm.weight/bias
        layer_norm.weight/bias
        embed_tokens.{k}.weight
        lm_heads.{k}.weight
        enc_to_dec_proj.weight/bias
    """
    # Strip decoder.model.decoder. prefix for transformer layers
    if k.startswith("decoder.model.decoder."):
        k = k[len("decoder.model.decoder.") :]
    # Strip decoder. prefix for lm_heads
    elif k.startswith("decoder.lm_heads."):
        k = k[len("decoder.") :]
    # enc_to_dec_proj stays as-is (already matches our attribute name)

    return k


def _load_weights(model_path: Path) -> dict[str, np.ndarray]:
    """Load model weights from safetensors or pytorch_model.bin.

    Prefers safetensors format. Falls back to pytorch_model.bin
    (requires torch: ``pip install mlx-audio-generate[convert]``).

    Handles both single-file and sharded (multi-file) weight storage.
    """
    import glob

    # 1. Single safetensors file
    sf_file = model_path / "model.safetensors"
    if sf_file.exists():
        return load_safetensors(sf_file)

    # 2. Sharded safetensors (model-00001-of-NNNNN.safetensors, ...)
    sf_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if sf_files:
        weights: dict[str, np.ndarray] = {}
        for sf in sf_files:
            shard = load_safetensors(sf)
            weights.update(shard)
            print(f"  Loaded {Path(sf).name}: {len(shard)} tensors")
        return weights

    # 3. Single pytorch_model.bin
    pt_file = model_path / "pytorch_model.bin"
    if pt_file.exists():
        print("  No safetensors found, loading pytorch_model.bin (requires torch)...")
        return load_pytorch_bin(pt_file)

    # 4. Sharded pytorch_model-NNNNN-of-NNNNN.bin
    pt_files = sorted(glob.glob(str(model_path / "pytorch_model-*.bin")))
    if pt_files:
        n = len(pt_files)
        print(f"  No safetensors, loading {n} pytorch shards...")
        weights = {}
        for pt in pt_files:
            shard = load_pytorch_bin(pt)
            weights.update(shard)
            print(f"  Loaded {Path(pt).name}: {len(shard)} tensors")
        return weights

    raise FileNotFoundError(
        f"No model weight files found in {model_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def convert_musicgen(
    repo_id: str,
    output_dir: str | Path,
    dtype: str | None = None,
) -> None:
    """Download and convert a MusicGen model from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID (e.g. 'facebook/musicgen-small').
        output_dir: Directory to write converted safetensors files.
        dtype: Optional cast dtype ('float16', 'bfloat16', 'float32').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from HF (include .bin files as fallback for repos without safetensors)
    print(f"Downloading {repo_id}...")
    model_path = download_model(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )

    # Load weights — prefer safetensors, fall back to pytorch_model.bin
    print("Loading weights...")
    weights = _load_weights(model_path)

    # Load HF config
    hf_config = {}
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            hf_config = json.load(f)

    # Sort weights into buckets
    t5_state: dict[str, np.ndarray] = {}
    decoder_state: dict[str, np.ndarray] = {}
    skipped_audio = 0
    skipped_other = 0

    total = len(weights)
    print(f"Processing {total} tensors...")

    for k, val in weights.items():
        # --- T5 Text Encoder ---
        if k.startswith("text_encoder."):
            clean_k = k[len("text_encoder.") :]
            clean_k = _remap_t5_key(clean_k)
            t5_state[clean_k] = val
            continue

        # --- Audio Encoder (EnCodec) --- SKIP
        # We load EnCodec separately from mlx-community
        if k.startswith("audio_encoder."):
            skipped_audio += 1
            continue

        # --- Decoder (transformer + embed_tokens + lm_heads) ---
        if k.startswith("decoder.model.decoder.") or k.startswith("decoder.lm_heads."):
            clean_k = _remap_decoder_key(k)
            decoder_state[clean_k] = val
            continue

        # --- enc_to_dec_proj (text) ---
        if k.startswith("enc_to_dec_proj."):
            decoder_state[k] = val
            continue

        # --- audio_enc_to_dec_proj (melody variant: chroma projection) ---
        if k.startswith("audio_enc_to_dec_proj."):
            decoder_state[k] = val
            continue

        # Anything else
        skipped_other += 1

    # Our T5EncoderModel has both shared.weight and encoder.embed_tokens.weight
    # pointing to the same embedding. HF only stores shared.weight, so we
    # duplicate it under both keys so load_weights(strict=True) succeeds.
    if "shared.weight" in t5_state and "encoder.embed_tokens.weight" not in t5_state:
        t5_state["encoder.embed_tokens.weight"] = t5_state["shared.weight"]

    # Optional dtype cast
    if dtype is not None:
        np_dtype = {
            "float16": np.float16,
            "bfloat16": np.float16,
            "float32": np.float32,
        }[dtype]
        for d in (t5_state, decoder_state):
            for k in d:
                if d[k].dtype in (np.float32, np.float64):
                    d[k] = d[k].astype(np_dtype)

    # Save split files
    print("\nConverted weights:")
    print(f"  T5 encoder:     {len(t5_state)} tensors")
    print(f"  Decoder:        {len(decoder_state)} tensors")
    print(f"  Audio encoder:  {skipped_audio} tensors (skipped, loaded separately)")
    if skipped_other:
        print(f"  Other skipped:  {skipped_other} tensors")

    save_safetensors(t5_state, output_dir / "t5.safetensors")
    save_safetensors(decoder_state, output_dir / "decoder.safetensors")

    # Save configs
    _save_configs(output_dir, hf_config, repo_id)

    # Save tokenizer
    _save_tokenizer(output_dir, repo_id)

    print(f"\nConversion complete! Weights saved to {output_dir}/")


def _save_configs(output_dir: Path, hf_config: dict, repo_id: str = "") -> None:
    """Write config.json and t5_config.json from HF's config."""
    # Detect melody variant and annotate config
    model_type = hf_config.get("model_type", "")
    if "melody" in model_type or "melody" in repo_id.lower():
        hf_config.setdefault("is_melody", True)
        hf_config.setdefault("num_chroma", 12)
        hf_config.setdefault("chroma_length", 235)

    # Save the full HF config as our config (from_dict handles the nesting)
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Extract T5 config separately for easy loading
    t5_section = hf_config.get("text_encoder", {})
    t5_config = {
        "d_model": t5_section.get("d_model", 768),
        "num_heads": t5_section.get("num_heads", 12),
        "d_kv": t5_section.get("d_kv", 64),
        "d_ff": t5_section.get("d_ff", 3072),
        "num_layers": t5_section.get("num_layers", 12),
        "vocab_size": t5_section.get("vocab_size", 32128),
        "relative_attention_num_buckets": t5_section.get(
            "relative_attention_num_buckets", 32
        ),
        "relative_attention_max_distance": t5_section.get(
            "relative_attention_max_distance", 128
        ),
    }
    with open(output_dir / "t5_config.json", "w") as f:
        json.dump(t5_config, f, indent=2)


def _save_tokenizer(output_dir: Path, repo_id: str) -> None:
    """Download and save the T5 tokenizer alongside the weights."""
    from transformers import AutoTokenizer

    # Check if already saved
    if (output_dir / "tokenizer_config.json").exists():
        return

    print("Downloading T5 tokenizer...")
    # MusicGen uses the t5-base tokenizer
    for source in [repo_id, "google-t5/t5-base"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)  # nosec B615 — known HF repo
            tokenizer.save_pretrained(str(output_dir))
            print(f"Saved tokenizer to {output_dir}")
            return
        except (OSError, ValueError, KeyError) as e:
            print(f"  Could not load tokenizer from {source}: {e}")
            continue

    print("Warning: Could not save tokenizer. It will be downloaded at runtime.")
