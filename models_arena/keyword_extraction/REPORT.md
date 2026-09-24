# Keyword Extraction Model Arena — Spotify App Store reviews

## Scope and frozen data

This is an offline comparison of four extractive approaches on the **existing** Spotify App Store evaluation set. No reviews were recollected, no sentiment labels were changed, and no sentiment model was retrained or benchmarked again. Of 391 preserved reviews, **255** have the saved gold sentiment `negative`: **177** in the multilingual evaluation set, **40** in the separate Ukrainian set, and **38** in the separate English set. These sets are disjoint. The review languages among the 255 are: bg 11, de 13, en 53, es 11, fi 3, fr 10, it 13, kk 4, lt 11, nl 14, oc 1, pl 13, pt 16, ro 1, ru 16, sk 1, sr 1, tr 9, uk 53, zh 1. Per-language results with 1–4 reviews are descriptive only.

The unchanged source file SHA-256 values are `reviews.jsonl`: `b5f02873c2b3e93735a15b3d975d1cbb251e5729f25e80d7352af49dcd183b27`; `sentiment_labels.jsonl`: `cc2cfd0aec5b553a957e975756ccbd61aabef3e4b40b69c3ac434a635d30eb5`.

### Gold keyword annotation

Three independent blind Codex GPT-6 annotation runs each read 85 `reviewId`, `title`, `text` records without rating, candidate predictions, model identity, or scores. They marked up to five meaningful source-grounded keyphrases in the original language, including negation where needed. A separate blind review examined all 22 ambiguous rows. Automated validation verified row coverage, order, phrase limits, exact source spans, and source offsets. A length-only quality check found the first pass too verbose; the annotators shortened phrases from the source text without seeing candidate outputs. The initial gold scoring was discarded. Final gold labels were frozen at SHA-256 `dd1cae6c84e5bbc6b23eb4950652be79e48161b5aece015cacbd536333985e5a` before **final** evaluation. No model-specific edits were made.

The 255 final labels contain **592** keyphrases: 512 have 1–3 words and 80 have four words where context or negation requires them. Twenty reviews have no defensible specific phrase. Twenty-two keyword labels carry `needsHumanReview`; three gold-negative reviews also have a saved uncertain sentiment label, kept in the main set and excluded in a sensitivity analysis. LLM labels remain provisional: dialect, irony, product attribution, morphology, and vague short reviews need a qualified human check before treating these values as definitive. The three sentiment-uncertain reviews are particularly unsuitable as unquestioned negative gold.

## Methods selected before evaluation

All three neural entries use the **same KeyBERT-style candidate-ranking method**, with different pretrained multilingual encoders; these are **not** three independent keyphrase extraction architectures. The method embeds the title-plus-body and each source n-gram, ranks by cosine relevance, and applies fixed maximal marginal relevance (0.75 relevance / 0.25 redundancy). This design keeps the text extractive and allows a controlled encoder comparison. [KeyBERT documentation](https://maartengr.github.io/KeyBERT/api/keybert.html) describes this family of embedding-based candidate ranking.

| Entry | Pinned checkpoint | Architecture and languages | Model-card license | Reason |
| --- | --- | --- | --- | --- |
| MiniLM | [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | 12-layer BERT, 384-dim sentence embeddings; model card says 50 languages | Apache-2.0 | Compact multilingual paraphrase encoder; useful baseline for short reviews. |
| E5-small | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), `614241f622f53c4eeff9890bdc4f31cfecc418b3` | 12-layer BERT, 384-dim embeddings; model card lists roughly 100 languages including Ukrainian | MIT | Multilingual retrieval encoder that could rank complaint phrases better; `query: ` is prefixed to both document and phrase as its model card directs for symmetric comparison. |
| BGE-M3 | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3), `5617a9f61b028005a4858fdac845db406aefb181` | 24-layer XLM-R, 1024-dim dense embeddings; more than 100 languages | MIT | Larger multilingual encoder that may preserve meaning in mixed and morphologically varied reviews; no query instruction required. |

The model-card license labels and language claims above are the publishers' statements, not an independent legal or per-language quality audit. We also considered [KBIR-Inspec](https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec), a genuine token-classification keyphrase model; its own card says it only works for English and is tuned to scientific abstracts. Substituting it for one multilingual candidate would not meet the Ukrainian review requirement. No checkpoint was fine-tuned on this evaluation set.

### Shared candidate extraction and TF-IDF

All four methods receive the same `prepare_review_text` copy of `title + text` (the collected review schema calls Apple's content field `text`). Unicode-aware word spans from Cyrillic, Latin, and other scripts yield 1–3 token candidates. Per-language stopwords from `stopwordsiso`, a small generic-app list, and explicit negation guards filter uninformative candidates. Phrases cannot cross sentence punctuation; up to 80 candidates per long review are sampled across the whole text. Parameters were fixed before the final evaluation and are identical for the three encoders. Neither ratings nor gold keywords enter candidate extraction.

TF-IDF uses scikit-learn `TfidfVectorizer` with Unicode words, 1–3 grams, sublinear term frequency, and IDF fitted **only to the 255 review texts**, with no gold keyword labels. It scores the same candidate phrases and returns up to five non-redundant phrases per review. Global IDF avoids empty per-language corpora for languages with one example; the language-specific candidate filter still removes many function words. Empty texts and empty vocabulary are handled as empty outputs.

## Evaluation protocol

Every approach predicts up to five ranked source phrases per review. Scores are cosine similarities for the neural methods and normalized TF-IDF weights for the baseline; they are not directly comparable across methods. Phrase matching applies Unicode NFC, case folding, punctuation and whitespace normalization. Exact normalized matches count; near matches require token Jaccard at least 0.8 with at least two shared words and identical explicit negation tokens. A bipartite one-to-one match prevents one prediction from satisfying multiple gold phrases. This deliberately conservative rule can miss true morphological paraphrases; the blind semantic audit below checks meaning separately without using embedding similarity as the only judge.

Per-review Precision@5 = matched predictions / number of predictions returned (at most five); Recall@5 = matched gold phrases / number of gold phrases; F1@5 is their harmonic mean. If both lists are empty, all three equal 1; if exactly one is empty, they equal 0. Reported subset values are **macro means over reviews**, not pooled phrase counts. The `all` group includes all 255 reviews; multilingual, Ukrainian and English are disjoint. Results excluding the three uncertain sentiment labels are reported separately. There is no development-set fitting or parameter search against gold labels.

## Measured results

All four methods returned results for **255/255** reviews without inference errors. Values below use the **final** frozen keyword labels; the preliminary scores produced before the length correction are discarded. In the first table each cell is **Precision / Recall / F1** at five, macro averaged by review.

| Method | All 255 | Multilingual 177 | Ukrainian 40 | English 38 |
| --- | --- | --- | --- | --- |
| TF-IDF | .091 / .224 / **.123** | .085 / .213 / **.116** | .135 / .347 / **.186** | .069 / .143 / **.090** |
| MiniLM | .124 / .292 / **.166** | .122 / .284 / **.162** | .170 / .426 / **.233** | .085 / .186 / **.112** |
| E5-small | .125 / .283 / **.165** | .123 / .275 / **.161** | .150 / .380 / **.207** | .106 / .218 / **.137** |
| BGE-M3 | .126 / .287 / **.166** | .117 / .269 / **.154** | .185 / .438 / **.250** | .106 / .212 / **.135** |

BGE-M3 has the highest overall and separate Ukrainian F1, but its overall lead over MiniLM is only **0.0008**. MiniLM has the highest multilingual-set F1; E5-small has the highest English F1 by only **0.002** over BGE-M3. These tiny differences should not be treated as statistically decisive. The low absolute scores reflect both missed complaint phrases and conservative phrase matching; the semantic review below is therefore material to the selection.

### Language breakdown within the multilingual set

The multilingual set contains English and Ukrainian reviews as well as other languages. The separate English and Ukrainian sets above are distinct. Values are macro **F1@5**; full per-language precision, recall, support and predictions are in each method's `metrics.json` and `predictions.jsonl`.

| Language | n | TF-IDF | MiniLM | E5-small | BGE-M3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bg | 11 | .180 | .190 | .216 | .198 |
| de | 13 | .107 | .107 | .140 | .123 |
| en | 15 | .017 | .030 | .031 | .017 |
| es | 11 | .261 | .275 | .275 | .269 |
| fi | 3 | .111 | .111 | .111 | .111 |
| fr | 10 | .134 | .176 | .176 | .106 |
| it | 13 | .156 | .269 | .195 | .267 |
| kk | 4 | .000 | .083 | .083 | .155 |
| lt | 11 | .113 | .275 | .267 | .198 |
| nl | 14 | .127 | .143 | .173 | .157 |
| oc | 1 | .000 | .000 | .000 | .000 |
| pl | 13 | .056 | .123 | .179 | .138 |
| pt | 16 | .090 | .118 | .119 | .105 |
| ro | 1 | .000 | .000 | .000 | .000 |
| ru | 16 | .184 | .207 | .225 | .212 |
| sk | 1 | .000 | .286 | .286 | .286 |
| sr | 1 | .000 | .000 | .000 | .000 |
| tr | 9 | .138 | .197 | .138 | .165 |
| uk | 13 | .078 | .145 | .091 | .139 |
| zh | 1 | .000 | .000 | .000 | .000 |

### Sensitivity to uncertain sentiment labels

Excluding the **three** previously flagged gold-negative sentiment rows changes the overall sample to **252**. F1@5 changes from .123 to **.125** for TF-IDF, .166 to **.167** for MiniLM, .165 to **.166** for E5-small, and .166 to **.167** for BGE-M3. The relative pattern is stable. This excludes uncertain *sentiment* labels only; the 22 keyword annotations flagged for human review remain in the main scores and are listed in `keyword_labels.jsonl` for follow-up.

### Local performance

Measurements were taken on this Windows CPU host after checkpoints had been downloaded to the local cache. `load` is checkpoint load time (TF-IDF: vectorizer fitting); `mean` is post-load inference time per review; RSS is process resident memory at the end of the run. These are indicative single-run values, not controlled hardware benchmarks.

| Method | Load / fit (s) | Total (s) | Mean inference (s/review) | RSS (MB) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | .29 | 1.17 | .005 | 128 |
| MiniLM | 32.77 | 152.29 | .469 | 790 |
| E5-small | 20.57 | 159.09 | .543 | 778 |
| BGE-M3 | 21.47 | 1320.07 | 5.093 | 1311 |

### Concrete review examples

The following are **actual saved predictions**, shown in rank order. The text excerpts are source review text; full records and scores are in `results/*/predictions.jsonl`.

**Ukrainian, `14434505174`** — “У застосунку зникають пісні? ... у мене вилучені з плейлиста ... ваш застосунок блокує.” Gold: `зникають пісні`; `вилучені з плейлиста`; `застосунок блокує`.

| Method | Five predicted phrases |
| --- | --- |
| TF-IDF | `зникають пісні`; `зникають`; `У мене`; `пісні`; `мене` |
| MiniLM | `застосунку зникають пісні`; `авторскі права, то`; `зникають пісні`; `ці пісні`; `ваш застосунок блокує` |
| E5-small | `застосунку зникають пісні`; `Якщо це авторскі`; `ваш застосунок блокує`; `ці пісні`; `вилучені з плейлиста` |
| BGE-M3 | `застосунку зникають пісні`; `Ютубмюзік`; `ваш застосунок блокує`; `зникають пісні`; `вилучені з плейлиста` |

BGE-M3 and E5-small recover two specific complaint spans; TF-IDF repeats generic fragments, and MiniLM also spends a slot on a broken copyright fragment. BGE-M3 has one redundant `зникають пісні` variant.

**English, `11403128585`** — complaints about Spotify adding unwanted songs to a playlist, playing only those songs, excessive ads, and six skips per hour. Gold: `songs I don’t want`; `plays ONLY them`; `too much ads`; `only 6 skips`.

| Method | Five predicted phrases |
| --- | --- |
| TF-IDF | `plays`; `playlist`; `songs`; `created a playlist`; `lot of songs` |
| MiniLM | `playlist Spotify`; `created a playlist`; `bit annoying`; `lot of songs`; `playlist` |
| E5-small | `created a playlist`; `annoying`; `ads`; `playlist Spotify`; `lot of songs` |
| BGE-M3 | `created a playlist`; `bit annoying`; `playlist Spotify`; `songs that aren’t`; `skips` |

All four miss much of the crucial unwanted-song and ad-frequency context. The shared candidate filter discards number-bearing spans such as `only 6 skips` and some stopword-edge phrases such as `too much ads`; this is a **shared extraction limitation**, not evidence that one encoder understands them. It was not tuned on the evaluation set after observing the results.

**Lithuanian, `11636201031`** — the review describes search returning different songs with the same name, too few skips, rewind difficulty, and many ads. Gold: `visiškai kitos dainos`; `tik kelis kartus`; `Persukti nevisada galima`; `LABAI daug reklamų`.

| Method | Five predicted phrases |
| --- | --- |
| TF-IDF | `dainos`; `anksčiau buvo geriau`; `dainas su TUO`; `galima ir pasidarė`; `išmeta į dainas` |
| MiniLM | `visiškai kitos dainos`; `Spotify” yra`; `Nepasakyčiau, kad “Spotify`; `LABAI daug reklamų`; `išmeta į dainas` |
| E5-small | `Nepasakyčiau, kad “Spotify`; `dainos pavadinimą`; `yra puiki programa`; `Spotify” yra puiki`; `LABAI daug reklamų` |
| BGE-M3 | `Nepasakyčiau, kad “Spotify`; `Spotify” yra puiki`; `LABAI daug reklamų`; `nesuprantu, kam reikėjo`; `paiešką dainos pavadinimą` |

E5-small and BGE-M3 can elevate the phrase `puiki programa` out of a negative context; the candidate-based approach does not reliably preserve negation across sentence fragments. MiniLM recovers two gold complaint spans here but also produces truncated fragments.

### Error patterns and limits

TF-IDF often promotes frequent or idiosyncratic tokens without enough context (`пісні`, `мене`, `plays`). All embedding methods sometimes select adjacent overlapping n-grams that restate one issue. MMR reduces but does not eliminate this. Candidate generation sometimes cuts across grammatical context, yielding `Spotify” yra` or `авторскі права, то`; it can omit important numbers, negation, inflected forms or four-word gold spans. English long reviews with several complaints are particularly difficult within five phrases. Ukrainian inflection and colloquial spelling produce both true semantic matches missed by conservative literal scoring and genuinely missed issues. The low-resource-language cells above are too small to justify language-specific routing.

The three methods and TF-IDF share source-bound extraction, so none invents a phrase absent from the normalized review text. However, an exact source fragment can still misrepresent the review by dropping negation or context. The separate blinded semantic audit tests this directly.

## Blind semantic relevance audit

A separate GPT-6 evaluator received **24 reviews**, sampled with seed `20260924`: eight from each evaluation set. For each review, the four five-phrase outputs were shuffled under anonymous letters; the evaluator saw only title, body, language and candidate phrases, **not** model identities, gold labels, ratings, or benchmark scores. It scored each group on a 0–5 utility scale and counted useful, source-supported phrases, meaning reversals, and generic/redundant phrases. The anonymous input, independent judgments and post-hoc unblinding key are saved under `results/`. This is a small, LLM-based secondary audit, not human adjudication or a replacement for the full-set metrics.

| Method | Mean utility / 5 | Useful phrase share | Meaning errors | Generic or redundant phrases | Utility: multi / uk / en |
| --- | ---: | ---: | ---: | ---: | --- |
| TF-IDF | 2.125 | .296 | 1 | 75 | 1.75 / 2.50 / 2.13 |
| MiniLM | 2.500 | .398 | 1 | 64 | 2.13 / 3.00 / 2.38 |
| E5-small | 2.500 | .398 | 3 | 62 | 2.50 / 2.38 / 2.63 |
| BGE-M3 | **2.667** | .389 | 3 | 63 | 2.25 / 3.00 / 2.75 |

BGE-M3 has the highest overall and English utility; it ties MiniLM on Ukrainian utility, while MiniLM and E5-small have a slightly higher share of individually useful phrases. E5-small has the highest utility in the eight sampled multilingual reviews. The auditor identified explicit meaning reversals, including BGE-M3's `бачу відміни` where a Ukrainian review says the cancellation option is **not** visible, and `споживаєш російський контент` where another review says the user **does not** consume it. This is a serious limitation for downstream issue metrics. The eight English and eight Ukrainian samples are too small to conclude that any difference of a few tenths is reliable.

## Selection and production integration

**Selected: BGE-M3 with the fixed KeyBERT-style extraction method**, pinned at `5617a9f61b028005a4858fdac845db406aefb181`. It has the highest full-set and separate Ukrainian F1@5, the highest overall blind utility, and the highest blind English utility. It was selected for measured quality despite taking about **5.09 s/review** on this CPU, roughly ten times MiniLM's inference time. MiniLM remains a credible lighter alternative because its multilingual F1 is higher and its blind negation-error count is lower, but the current evidence does not justify automatic language routing. All three model predictions and TF-IDF predictions remain saved for later experiments. No sentiment checkpoint or prior arena result is changed.

The production `KeywordExtractor` in `appstore_reviews/keywords.py` defaults to this exact checkpoint. It accepts one review or a list, uses the saved `language`, returns up to five source phrases with cosine scores and review ID, lazily loads and reuses one model instance, and reports malformed text or per-review inference errors. The module does not use ratings, gold labels, Model Arena files, or sentiment as a condition. `analyze_negative_keywords` in `appstore_reviews/pipeline.py` runs the existing Cardiff sentiment analyzer, filters its negative results, passes those reviews to the same keyword extractor, and returns per-review sentiment and keyword records plus `common_keywords_by_language`. The latter counts unique review IDs, share, mean score and examples, keeping languages separate. Aggregation deliberately merges only surface case/punctuation/spacing variants; more aggressive stemming is deferred because merging different complaints would distort frequency metrics.

The [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) declares MIT. This experimental integration uses its dense embedding through SentenceTransformers, not its sparse/ColBERT heads. Model-card licensing should be verified for a commercial deployment. The **overall pipeline** still uses Cardiff sentiment, whose model card lacks a confirmed commercial license as documented in the [sentiment arena report](../sentiment/REPORT.md); this keyword change does not resolve that question.

## Reproduce and tests

From the repository root, install `python -m pip install -e ".[sentiment,keywords]"` and `python -m pip install pytest`; then run:

```powershell
python -m models_arena.keyword_extraction.download_models
python -m models_arena.keyword_extraction.benchmark
python -m pytest -q
```

The first command downloads the pinned checkpoints to the standard Hugging Face cache; weights are not committed. The benchmark uses the saved 255 gold-negative reviews and verifies the final gold-label SHA-256 at start and end. Set `KEYWORD_ARENA_EVALUATE_ONLY=1` to recompute metrics from saved predictions without inference. Detailed `comparison.json`, method `metrics.json` files, all 255 predictions per method, semantic audit artifacts, and a real production smoke result are preserved in `results/`.

The unit suite checks Unicode extraction, single and batch APIs, empty/malformed input, model reuse, one-to-one negation-aware matching, gold source offsets, empty TF-IDF vocabulary, unique-review aggregation, and sentiment-negative filtering. The existing scraper and sentiment tests also run. On the staged project, **31/31 tests passed**. A separate real checkpoint smoke loaded pinned BGE-M3, extracted phrases from one actual Ukrainian and one actual English saved review, reused the same model instance, and ran the full Cardiff → negative filter → BGE pipeline on three saved reviews. Cardiff classified the third review positive and both complaint reviews negative; two negative reviews entered keyword extraction, with no recorded error. Its compact actual outputs are in `results/production_smoke.json`. This was inference, not a mock or rerun of the sentiment benchmark.

### Remaining limitations

This is one app's public RSS sample with LLM reference labels, no human-adjudicated keyword gold or cross-app generalization study. Several languages have 1–4 examples. Very low absolute F1 and the blind audit's meaning reversals show that extracted phrases are unsuitable as definitive issue counts without human inspection. The shared candidate stage can miss numeric restrictions, edge stopwords, long phrases and morphology; the chosen encoder cannot recover candidates it never receives. Common phrase counts are surface-level frequencies, not automatically merged semantic issue categories. A separate evaluation set would be needed before tuning extraction rules, thresholds, or language routing without leakage into these benchmark numbers.
