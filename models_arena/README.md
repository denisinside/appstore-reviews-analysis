# Model Arena

This directory holds offline NLP experiments for sentiment and keyword extraction. The review collector in `appstore_reviews` remains independent of these evaluation files.

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

### Candidate and ranking revision

The original Spotify benchmark and its `results/` are retained as the v1 record. A blind review of its 22 uncertain annotations produced `keyword_label_audit.jsonl`, eight changes in `keyword_label_changes.jsonl`, and frozen `keyword_labels_v2.jsonl`; the original labels remain untouched.

The v2 comparison uses the three pinned encoders—E5-small, BGE-M3, and MiniLM—and TF-IDF with revised shared phrase generation and ranking. Production keyword extraction uses [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), pinned to revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; v2 applies the `query: ` prefix to both review text and candidate phrases. Run one method at a time from the repository root:

```powershell
python -m models_arena.keyword_extraction.benchmark_v2 --dataset old --model minilm
python -m models_arena.keyword_extraction.benchmark_v2 --dataset heldout --model minilm
```

Replace `minilm` with `e5_small`, `bge_m3`, or `tfidf`. `old` is a regression dataset that informed the changes. The separate `heldout_crossapp_v1` data comes from real Duolingo and Telegram RSS reviews; its independent annotations were frozen before final inference. Scores and predictions are written to `keyword_extraction/results_v2/` and `keyword_extraction/results_heldout_v1/`, preserving v1 outputs. These saved quality scores precede the final local phrase and evidence fixes; final production F1 remains unconfirmed. Negative filtering in the production pipeline selects reviews; it does not assign negative labels to each extracted phrase. Extracted terms are source keyphrases, not confirmed product issues, so inspect evidence and original review text before interpreting counts. See the appended v2 section of the [report](keyword_extraction/REPORT.md) for the evaluation protocol and limitations. The Cardiff sentiment checkpoint used for review filtering has no declared model-card license; commercial permission is unconfirmed.

Running `benchmark_v2` again with current production code can overwrite the archived prediction files with outputs from the later phrase fixes. Copy the result folders before any new experiment if you need to preserve the frozen comparison.
