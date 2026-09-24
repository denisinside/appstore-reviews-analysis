"""Download the three pinned checkpoints into the Hugging Face cache once."""

from __future__ import annotations

import os

from huggingface_hub import file_download, snapshot_download

from .benchmark import MODELS


def main() -> None:
    if os.name == "nt":
        # Windows accounts without Developer Mode cannot create cache symlinks.
        file_download.are_symlinks_supported = lambda cache_dir=None: False
    for model_id, revision, _ in MODELS.values():
        print(f"Downloading {model_id} @ {revision}", flush=True)
        snapshot_download(repo_id=model_id, revision=revision,
                          allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                                          "sentence_bert_config.json", "1_Pooling/config.json",
                                          "model.safetensors", "pytorch_model.bin", "tokenizer.json",
                                          "tokenizer_config.json", "special_tokens_map.json",
                                          "sentencepiece.bpe.model", "unigram.json", "vocab.txt"],
                          max_workers=1)


if __name__ == "__main__":
    main()
