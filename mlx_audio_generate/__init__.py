import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from mlx_audio_generate.version import __version__

__all__ = ["__version__"]
