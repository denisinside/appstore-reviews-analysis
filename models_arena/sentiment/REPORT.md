# Spotify App Store sentiment Model Arena

Run date: 2026-09-24. This report describes actual inference on saved public App Store reviews. Full per-review predictions, class precision/recall/F1, confusion matrices, language breakdowns and timing are in each model's `results/<model>/metrics.json` and `predictions.jsonl`; `results/comparison.json` combines them.

## Dataset and reference labels

The existing parser's Discovery ranked `fr, bg, nl, at, it, ie, lt, pl, pt, kz` as Spotify's top ten supported storefronts. A one-off Top collection returned **4,939 unique reviews**. A separate Ukrainian storefront collection returned **498**. From these source snapshots, the frozen `spotify_appstore_v1` contains **391 distinct review IDs and 391 distinct normalized texts**: 291 in the multilingual sample, plus 50 separate Ukrainian and 50 separate English texts. The dataset has 70 `uk` and 70 `en` fastText-tagged reviews overall. The 391 rows come from eleven storefronts (`at, bg, fr, ie, it, kz, lt, nl, pl, pt, ua`). The original titles and texts are preserved; the benchmark applies the existing `prepare_review_text` only to its input copy.

Thirty-three language codes appeared in the top-ten collection. The multilingual sample contains 31 of them. The `ie` and `tg` candidates failed the minimum text/confidence checks. Language shortages are listed in `datasets/spotify_appstore_v1/metadata.json`; 20 examples were available for each of `bg, de, en, es, fr, it, lt, nl, pl, pt, ru, uk`, 18 for `tr`, and only 1–4 for many others. These small per-language scores are descriptive, not reliable estimates of general accuracy. fastText labels are automatic; some short or mixed-language reviews visibly have the wrong language code.

GPT-6 Luna agents annotated the title and body of every row without seeing ratings, storefronts, model predictions or other metadata. The label means overall sentiment toward Spotify or the user's experience. A second blinded pass revisited 73 ambiguous, low-confidence or flagged rows and changed 19 annotations or flags **before** the model benchmark. The final reference has **255 negative, 127 positive, 9 neutral** labels; 71 are marked ambiguous and 13 still need human review. The reference is provisional: it is AI annotation, has few neutral cases, and some labels may be wrong. Model disagreement is not proof of a bad model. The source files, selection script, reference labels, change log, and hashes make this run auditable. The saved v1 set was not altered in response to model results.

## Models and conditions

| Model | Architecture and label mapping | Checkpoint revision | Published license |
|---|---|---|---|
| [Cardiff XLM-R sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | XLM-RoBERTa sequence classifier, negative/neutral/positive | `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8` | No license stated on the checkpoint card/API; commercial rights are unclear |
| [Tabularis multilingual sentiment](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) | DistilBERT sequence classifier; sum Very Negative+Negative and Positive+Very Positive probabilities, keep Neutral | `eea032081f8d247b4303ef3565e7cec1b6f201c9` | CC BY-NC 4.0; commercial use requires separate rights |
| [lxyuan multilingual DistilBERT](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student) | DistilBERT sequence classifier, configured positive/neutral/negative labels | `cf991100d706c13c0a080c097134c05b7f436c45` | Apache 2.0 |

All three official pretrained classification checkpoints were downloaded and run locally; no fine-tuning occurred. The same frozen review IDs, order, normalized title+body text, batch size 8, truncation at 256 tokens, CPU device and four PyTorch threads were used. A single warmup was excluded from timing. Environment: Windows 11, AMD64 Family 25 Model 80 CPU, Python 3.12, PyTorch 2.14.0, Transformers 4.57.6, Hugging Face Hub 0.36.2, Tokenizers 0.22.2. `inference_seconds` includes tokenization and model forward passes; `load_seconds` is separate. Macro-F1 always averages all three sentiment classes, including zero F1 for a class with no examples in a tiny language. Every model returned **391/391** predictions with no inference errors.

## Primary results

| Model | Multilingual accuracy / Macro-F1 | Ukrainian accuracy / Macro-F1 | English accuracy / Macro-F1 | Negative recall multi / uk / en | Mean language Macro-F1 | Load / inference seconds | Mean inference ms/review |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cardiff XLM-R | 0.722 / **0.523** | 0.800 / **0.638** | 0.740 / 0.455 | 0.667 / **0.775** / **0.842** | **0.387** | 2.11 / 44.61 | 114.1 |
| Tabularis | 0.663 / 0.490 | 0.680 / 0.472 | 0.740 / **0.556** | **0.672** / 0.650 / 0.789 | 0.330 | 0.34 / 28.61 | 73.2 |
| lxyuan DistilBERT | 0.632 / 0.429 | 0.660 / 0.476 | 0.520 / 0.355 | 0.554 / 0.700 / 0.474 | 0.329 | 0.39 / 26.12 | 66.8 |

The multilingual sample has 291 reviews, Ukrainian 50 and English 50. The best multilingual and Ukrainian quality is Cardiff; Tabularis leads English Macro-F1. Cardiff also has the highest negative-class recall on the Ukrainian and English samples. lxyuan has materially worse English negative recall (18/38) and remains only an arena baseline. In all three models, neutral scores are unstable because the reference contains only nine neutral examples. Full per-class precision, recall, F1 and confusion matrices are in the JSON result files.

## Multilingual results by fastText language

Macro-F1 is shown for the multilingual subset only. The number `n` is identical across models. Values at very small `n` are especially fragile.

| Language | n | Cardiff | Tabularis | lxyuan |
|---|---:|---:|---:|---:|
| ar | 1 | 0.000 | 0.000 | 0.000 |
| bg | 20 | 0.536 | 0.481 | 0.476 |
| ceb | 1 | 0.333 | 0.333 | 0.333 |
| cs | 2 | 0.333 | 0.333 | 0.222 |
| de | 20 | 0.513 | 0.431 | 0.467 |
| en | 20 | 0.524 | 0.619 | 0.333 |
| eo | 2 | 0.333 | 0.333 | 0.222 |
| es | 20 | 0.582 | 0.541 | 0.439 |
| fi | 4 | 0.600 | 0.222 | 0.389 |
| fr | 20 | 0.491 | 0.357 | 0.418 |
| gl | 2 | 0.333 | 0.222 | 0.333 |
| he | 1 | 0.333 | 0.333 | 0.333 |
| hu | 2 | 0.333 | 0.333 | 0.333 |
| it | 20 | 0.576 | 0.437 | 0.397 |
| ja | 2 | 0.222 | 0.333 | 0.333 |
| kk | 4 | 0.222 | 0.222 | 0.286 |
| ky | 1 | 0.333 | 0.000 | 0.333 |
| lt | 20 | 0.392 | 0.583 | 0.282 |
| mk | 1 | 0.333 | 0.000 | 0.333 |
| nl | 20 | 0.579 | 0.472 | 0.430 |
| no | 1 | 0.000 | 0.000 | 0.333 |
| oc | 1 | 0.000 | 0.000 | 0.000 |
| pl | 20 | 0.520 | 0.398 | 0.487 |
| pt | 20 | 0.395 | 0.447 | 0.385 |
| ro | 2 | 0.667 | 0.667 | 0.667 |
| ru | 20 | 0.431 | 0.499 | 0.374 |
| sk | 1 | 0.000 | 0.000 | 0.000 |
| sr | 2 | 0.667 | 0.333 | 0.222 |
| tr | 18 | 0.547 | 0.503 | 0.442 |
| uk | 20 | 0.585 | 0.456 | 0.427 |
| zh | 3 | 0.267 | 0.333 | 0.167 |

## Error review

`results/error_examples.json` retains the original title/body, reference and all three predictions for **220** reviews where at least one model disagreed with the reference; all three disagreed on **43**. The following examples show distinct failure modes:

| Review ID | Language | Reference | Cardiff / Tabularis / lxyuan | Observation |
|---|---|---|---|---|
| `14289198825` | uk | negative | neutral / positive / positive | “Багато реклами! Ліміт перемикання пісень! Все платно!” is an obvious complaint missed by every model. |
| `14090252525` | uk | negative | neutral / neutral / positive | CarPlay displays a different song after an update; issue description lacks strong polarity words. |
| `13877772672` | en | negative | neutral / negative / negative | Cardiff misses a clear complaint that ads doubled. |
| `14001948936` | en | negative | negative / neutral / negative | Tabularis treats a shuffle complaint as neutral. |
| `13303806366` | en | negative | negative / negative / positive | lxyuan marks an “AI generated trash” complaint positive. |
| `14302432115` | sk | negative | positive / positive / positive | Sarcastic thanks about artist payouts fools all models. |
| `14537698598` | en | negative | neutral / positive / positive | The text is mostly explicit praise, so this reference label itself needs human audit. |
| `14358457646` | en | negative | positive / positive / positive | “good … BRILLIANT” with an ad caveat may also be a reference error. |

The final two rows remain unchanged in v1; model votes were never used to rewrite reference labels. They illustrate why the AI-only reference, mixed sentiment and irony need human adjudication before claiming general production quality. Another language issue: review `11981682744` is German text about ads but fastText tagged it `nl`, so its per-language placement is unreliable.

## Selection and integration

**Cardiff XLM-RoBERTa** is integrated as the optional experimental analysis step because it led on multilingual and Ukrainian Macro-F1 and on Ukrainian and English negative-class recall. The integration pins the exact benchmark revision `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`; speed and checkpoint size did not drive the choice. Cardiff's model card/API does not state a license, so commercial-use permission is **not confirmed**. This experimental integration should not be treated as clearance for commercial deployment. Tabularis remains CC BY-NC 4.0 and lxyuan remains Apache 2.0; all three models and benchmark results remain in Model Arena for future comparison. Human review of unclear labels and further error testing are still advisable before using automated sentiment to drive product decisions.

The main package exposes `SentimentAnalyzer` from `appstore_reviews.sentiment`. It lazily loads the pinned model once, joins normalized title and body, uses batch inference, truncates long inputs, returns a separate `review_id`/model/three-class-score/result record, and reports empty text or per-review errors. Example:

```python
from appstore_reviews import SentimentAnalyzer

analyzer = SentimentAnalyzer(device="cpu")
results = analyzer.analyze_reviews(collected["reviews"])
```

The parser's `Country`, `Top`, `Discovery`, preprocessing and statistical metrics logic remain separate. The evaluation set is not loaded by this module. Offline parser and sentiment unit tests passed; an actual cached Cardiff checkpoint smoke test classified Ukrainian and English reviews, handled empty and long text, and confirmed that one model object was reused. Model weights are kept outside Git.
