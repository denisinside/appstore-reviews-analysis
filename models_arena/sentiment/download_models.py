"""Pin and download the three official pretrained sentiment checkpoints."""

import os

from huggingface_hub import snapshot_download
from huggingface_hub import file_download

# Windows without Developer Mode cannot create the cache's pointer symlinks.
# Force Hugging Face's documented copy fallback for this one-off download.
if os.name == "nt":
    file_download.are_symlinks_supported = lambda cache_dir=None: False

MODELS = [
    ("cardiffnlp/twitter-xlm-roberta-base-sentiment", "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8", "pytorch_model.bin"),
    ("tabularisai/multilingual-sentiment-analysis", "eea032081f8d247b4303ef3565e7cec1b6f201c9", "model.safetensors"),
    ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "cf991100d706c13c0a080c097134c05b7f436c45", "model.safetensors"),
]

for model_id, revision, weight in MODELS:
    print(f"Downloading {model_id} @ {revision}", flush=True)
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=[
            "config.json", weight, "tokenizer*", "vocab*", "sentencepiece*", "spiece*",
            "merges.txt", "special_tokens_map.json", "added_tokens.json",
        ],
        max_workers=1,
    )
    print(path, flush=True)
