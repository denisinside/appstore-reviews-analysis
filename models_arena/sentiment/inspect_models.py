import json
from huggingface_hub import HfApi

MODELS = [
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "tabularisai/multilingual-sentiment-analysis",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
]

api = HfApi()
for model_id in MODELS:
    info = api.model_info(model_id, files_metadata=True)
    print(json.dumps({
        "model_id": model_id,
        "sha": info.sha,
        "license": (info.cardData or {}).get("license"),
        "files": [{"name": item.rfilename, "size": item.size} for item in info.siblings if item.rfilename.endswith(("safetensors", "bin", "config.json"))],
    }, ensure_ascii=False))
