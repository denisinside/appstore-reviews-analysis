# Model Arena

This directory holds offline NLP experiments. The review collector in `appstore_reviews` remains independent of these evaluation files. The current experiment is sentiment classification of the frozen Spotify App Store dataset.

From the repository root, create an environment and install the optional model dependencies:

```powershell
python -m pip install -e ".[sentiment]"
python -m pip install sentencepiece
python models_arena/sentiment/download_models.py
python -m models_arena.sentiment.benchmark --model xlm_roberta
python -m models_arena.sentiment.benchmark --model tabularis_multilingual
python -m models_arena.sentiment.benchmark --model multilingual_distilbert
python -m models_arena.sentiment.compare
```

The download step fetches pinned weights into the standard Hugging Face cache; the benchmark then runs offline against saved dataset files. Results go into each model's own `results` folder. `sentiment/REPORT.md` explains the comparison and the production selection.

The arena code does not use review ratings as model input and does not retrain on the evaluation set.

## Keyword extraction

The [keyword extraction report](keyword_extraction/REPORT.md) compares three pinned multilingual embedding checkpoints used with one KeyBERT-style ranking method, plus a TF-IDF baseline, on the frozen 255 gold-negative Spotify reviews. Existing review and sentiment files are unchanged. New blinded keyword labels are in `datasets/spotify_appstore_v1/keyword_labels.jsonl`.

```powershell
python -m pip install -e ".[keywords]"
python -m models_arena.keyword_extraction.download_models
python -m models_arena.keyword_extraction.benchmark
```

The download step stores weights in the Hugging Face cache, outside Git. The benchmark writes complete per-review predictions, per-method metrics, and `results/comparison.json`. To score already saved predictions again without inference, set `KEYWORD_ARENA_EVALUATE_ONLY=1` before running the benchmark. The gold labels are protected by a recorded SHA-256 check.
