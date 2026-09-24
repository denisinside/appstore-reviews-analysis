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
