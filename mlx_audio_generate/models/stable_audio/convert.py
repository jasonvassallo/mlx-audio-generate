"""Convert Stable Audio Open weights from HuggingFace safetensors to MLX format.

Reads the original model.safetensors and splits it into:
    - vae.safetensors          (Oobleck VAE encoder/decoder)
    - dit.safetensors          (Diffusion Transformer)
    - t5.safetensors           (T5 text encoder)
    - conditioners.safetensors (time embedder raw weights)
    - config.json              (model configuration)
    - t5_config.json           (T5 configuration)

Weight conversions applied:
    - Conv1d:          PyTorch (Out, In, K) -> MLX (Out, K, In)
    - ConvTranspose1d: PyTorch (In, Out, K) -> MLX (Out, K, In)
    - Weight norm:     g, v pairs fused into w = g * (v / ||v||)
    - VAE key cleanup: strip 'pretransform.model.' prefix, add extra 'layers.'
    - DiT key cleanup: strip 'model.model.' prefix, rename transformer layers
"""

import json
from pathlib import Path

import numpy as np

from mlx_audio_generate.shared.hub import (
    download_model,
    load_safetensors,
    save_safetensors,
)
from mlx_audio_generate.shared.mlx_utils import (
    fuse_weight_norm,
    transpose_conv1d_weight,
    transpose_conv_transpose1d_weight,
)


def _convert_vae_key(k: str) -> str:
    """Convert VAE key from HF format to our MLX module format.

    HF:  pretransform.model.encoder.layers.0.layers.1.layers.0.weight
    MLX: encoder.layers.layers.0.layers.layers.1.layers.layers.0.weight

    The extra 'layers.' insertion is needed because MLX nn.Sequential
    stores children under a 'layers' attribute.
    """
    k = k.replace("pretransform.model.", "")
    parts = k.split(".")
    new_parts = []
    for i, part in enumerate(parts):
        new_parts.append(part)
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            new_parts.append("layers")
    return ".".join(new_parts)


def _is_vae_conv_transpose(key: str) -> bool:
    """Detect ConvTranspose1d layers in the VAE decoder.

    ConvTranspose layers are at decoder.layers.X.layers.1 (the upsampling conv
    inside each DecoderBlock). The key pattern has exactly 8 parts:
        pretransform.model.decoder.layers.X.layers.1.weight
    """
    parts = key.split(".")
    return (
        len(parts) == 8
        and parts[0] == "pretransform"
        and parts[1] == "model"
        and parts[2] == "decoder"
        and parts[3] == "layers"
        and parts[5] == "layers"
        and parts[6] == "1"
        and parts[7] == "weight"
    )


def _is_wn_conv_transpose(prefix: str) -> bool:
    """Detect ConvTranspose1d in weight-norm pairs by prefix.

    Prefix: pretransform.model.decoder.layers.X.layers.1
    7 parts, ending with decoder.layers.X.layers.1
    """
    parts = prefix.split(".")
    return (
        len(parts) == 7
        and parts[2] == "decoder"
        and parts[5] == "layers"
        and parts[6] == "1"
    )


def convert_stable_audio(
    repo_id: str,
    output_dir: str | Path,
    dtype: str | None = None,
) -> None:
    """Download and convert a Stable Audio Open model.

    Args:
        repo_id: HuggingFace repo ID (e.g. 'stabilityai/stable-audio-open-small').
        output_dir: Directory to write converted safetensors files.
        dtype: Optional cast dtype ('float16', 'bfloat16', 'float32').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from HF
    print(f"Downloading {repo_id}...")
    model_path = download_model(repo_id)

    # Load all safetensors from the snapshot
    print("Loading weights...")
    weights = load_safetensors(model_path / "model.safetensors")

    # Also try model_config.json for config info
    hf_config = {}
    config_file = model_path / "model_config.json"
    if config_file.exists():
        with open(config_file) as f:
            hf_config = json.load(f)

    # Sort weights into buckets
    vae_state: dict[str, np.ndarray] = {}
    dit_state: dict[str, np.ndarray] = {}
    cond_state: dict[str, np.ndarray] = {}
    t5_state: dict[str, np.ndarray] = {}
    wn_pairs: dict[str, dict[str, np.ndarray]] = {}

    total = len(weights)
    print(f"Processing {total} tensors...")

    for k, val in weights.items():
        # --- T5 Text Encoder ---
        if (
            "encoder.block" in k
            or (k.startswith("shared") and "embed" in k)
            or "final_layer_norm" in k
        ):
            t5_state[k] = val
            continue

        # --- VAE (Autoencoder) ---
        if k.startswith("pretransform."):
            # Weight normalization: collect g/v pairs
            if k.endswith(".weight_g"):
                prefix = k[:-9]  # strip '.weight_g'
                wn_pairs.setdefault(prefix, {})["g"] = val
                continue
            if k.endswith(".weight_v"):
                prefix = k[:-9]
                wn_pairs.setdefault(prefix, {})["v"] = val
                continue

            clean_k = _convert_vae_key(k)

            # Transpose conv weights
            if "weight" in k and val.ndim == 3:
                if _is_vae_conv_transpose(k):
                    val = transpose_conv_transpose1d_weight(val)
                else:
                    val = transpose_conv1d_weight(val)

            vae_state[clean_k] = val
            continue

        # --- DiT (Diffusion Transformer) ---
        if k.startswith("model.model."):
            clean_k = k.replace("model.model.transformer.layers.", "blocks.")
            clean_k = clean_k.replace("model.model.", "")

            if "conv" in k and "weight" in k and val.ndim == 3:
                val = transpose_conv1d_weight(val)

            dit_state[clean_k] = val
            continue

        # --- Conditioners ---
        if k.startswith("conditioner."):
            cond_state[k] = val
            continue

    # Fuse weight-norm pairs for VAE
    for prefix, pair in wn_pairs.items():
        if "g" in pair and "v" in pair:
            is_transpose = _is_wn_conv_transpose(prefix)
            w = fuse_weight_norm(pair["g"], pair["v"])

            if is_transpose:
                w = transpose_conv_transpose1d_weight(w)
            else:
                w = transpose_conv1d_weight(w)

            clean_prefix = _convert_vae_key(prefix)
            vae_state[clean_prefix + ".weight"] = w
        else:
            print(f"Warning: incomplete weight-norm pair for {prefix}")

    # Optional dtype cast
    if dtype is not None:
        np_dtype = {
            "float16": np.float16,
            "bfloat16": np.float16,
            "float32": np.float32,
        }[dtype]
        for d in (vae_state, dit_state, cond_state, t5_state):
            for k in d:
                if d[k].dtype in (np.float32, np.float64):
                    d[k] = d[k].astype(np_dtype)

    # Save split files
    print("\nConverted weights:")
    print(f"  VAE:          {len(vae_state)} tensors")
    print(f"  DiT:          {len(dit_state)} tensors")
    print(f"  Conditioners: {len(cond_state)} tensors")
    print(f"  T5:           {len(t5_state)} tensors")

    save_safetensors(vae_state, output_dir / "vae.safetensors")
    save_safetensors(dit_state, output_dir / "dit.safetensors")
    save_safetensors(cond_state, output_dir / "conditioners.safetensors")

    if t5_state:
        save_safetensors(t5_state, output_dir / "t5.safetensors")
    else:
        print("Warning: No T5 weights found. Downloading from stable-audio-open-1.0...")
        _download_t5_weights(output_dir)

    # Save configs
    _save_configs(output_dir, hf_config)

    # Save tokenizer
    _save_tokenizer(output_dir, repo_id)

    print(f"\nConversion complete! Weights saved to {output_dir}/")


def _download_t5_weights(output_dir: Path) -> None:
    """Download T5 encoder weights from stable-audio-open-1.0."""
    try:
        t5_path = download_model(
            "stabilityai/stable-audio-open-1.0",
            allow_patterns=["text_encoder/model.safetensors"],
        )
        t5_sf = t5_path / "text_encoder" / "model.safetensors"
        if t5_sf.exists():
            import shutil

            shutil.copy2(t5_sf, output_dir / "t5.safetensors")
            print("Downloaded T5 encoder weights.")
        else:
            print("Warning: Could not find T5 weights in download.")
    except Exception as e:
        print(f"Warning: Could not download T5 weights: {e}")


def _save_configs(output_dir: Path, hf_config: dict) -> None:
    """Write config.json and t5_config.json."""
    # Model config with sensible defaults for stable-audio-open-small
    config = {
        "dit": {
            "io_channels": 64,
            "embed_dim": 1024,
            "depth": 16,
            "num_heads": 8,
            "cond_token_dim": 768,
            "global_cond_dim": 768,
            "project_cond_tokens": True,
            "timestep_features_dim": 256,
        },
        "vae": {
            "in_channels": 2,
            "channels": 128,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8],
            "latent_dim": 64,
        },
        "sample_rate": hf_config.get("sample_rate", 44100),
        "sample_size": hf_config.get("sample_size", 524288),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # T5 config
    t5_config = {
        "d_model": 768,
        "num_heads": 12,
        "d_kv": 64,
        "d_ff": 3072,
        "num_layers": 12,
        "vocab_size": 32128,
    }
    with open(output_dir / "t5_config.json", "w") as f:
        json.dump(t5_config, f, indent=2)


def _save_tokenizer(output_dir: Path, repo_id: str) -> None:
    """Download and save the tokenizer alongside the weights."""
    from transformers import AutoTokenizer

    # Check if tokenizer already saved
    if (output_dir / "tokenizer_config.json").exists():
        return

    print("Downloading tokenizer...")
    for source in [repo_id, "stabilityai/stable-audio-open-1.0"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer")
            tokenizer.save_pretrained(str(output_dir))
            print(f"Saved tokenizer to {output_dir}")
            return
        except Exception:
            continue

    # Fallback to plain T5 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        tokenizer.save_pretrained(str(output_dir))
        print("Saved T5-base tokenizer as fallback.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
