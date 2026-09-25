# Keyword Extraction Model Arena — Spotify App Store reviews

Current production selection: **E5-small**, pinned to `614241f622f53c4eeff9890bdc4f31cfecc418b3`. The early BGE-M3 selection below is the preserved v1 historical result; the [final production integration](#final-production-integration-after-the-v2-comparison) section supersedes it.

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

---

## Follow-up v2: phrase quality and independent cross-app evaluation

The sections above are the **unaltered v1 record**. Their scores, predictions, model list and experimental conclusion remain available under `results/`. This follow-up responds to the observed incomplete phrases, duplicated keywords, lost negation and omitted numeric restrictions. It does not retrain any checkpoint, change Cardiff sentiment, recollect the original Spotify dataset, or implement Issue Extraction. All three pinned embedding checkpoints and TF-IDF remain in the arena.

### Gold audit and data separation

An independent reviewer re-examined the **22** Spotify keyword annotations previously marked `needsHumanReview` using only the source review and existing labels, without model predictions or scores. Fourteen were kept and eight revised. The original `keyword_labels.jsonl` remains at SHA-256 `dd1cae6c84e5bbc6b23eb4950652be79e48161b5aece015cacbd536333985e5a`. `keyword_label_audit.jsonl` records all 22 decisions; `keyword_label_changes.jsonl` stores before, after and reasons for eight revisions; `keyword_labels_v2.jsonl` is frozen at SHA-256 `44cb6c1f5ce5ca9396ef3399ea0bab49afe078d53958df17e1a33e7d88159465`. The revised labels contain 599 phrases, with 17 still marked uncertain. Some changed labels restore account blocking, unwanted parental control and children's media access, ad frequency and required viewing, repeated rating prompts, and explicitly qualified ethical complaints. These are reviewer judgments, not proven product facts.

The independent final set was collected with the existing public Apple RSS parser from **Duolingo** (`570060128`) and **Telegram** (`686449807`). Six app/storefront combinations (`ua`, `us`, `gb`, `pl` as available) each checked pages 1–10 without request errors; the collection had 2,800 accessible unique reviews across those combinations. From them, a deterministic seed selected 200 new reviews with ratings 1–2 and at least 25 title/body characters, balanced toward Ukrainian and English. Ratings were a **sampling filter only**; neither annotators nor keyword models received them as input. Two separate blind annotators each labeled 100 title/body records for sentiment and up to five source-grounded complaint phrases without seeing model output. The labels and source-span offsets were validated and frozen before final inference. Their 200 sentiment labels were 197 negative, two neutral and one positive. Two initial Ukrainian label/source rows were accidentally viewed by the implementation agent during setup; they were excluded from all final scoring and the semantic sample before any inspection of their model predictions. The resulting **195** gold-negative evaluation reviews comprise 77 Ukrainian, 80 English, 18 Polish and 20 Russian, with 398 gold keyphrases; 30 reviews retain uncertain keyword labels. The source records, raw independent annotations, frozen `keyword_labels.jsonl`, hashes and collection provenance are under `datasets/heldout_crossapp_v1/`. The frozen label SHA-256 is `3329280c786fff85ccbf37ced681c54d198214f1e77a0a119b255e0ace589307`; there are zero review-ID overlaps with Spotify.

This new set is **rating-enriched**, not a representative sample of complaint prevalence. The labels are independent of model predictions but are still LLM judgments, not qualified human adjudication. Results support comparison among these methods on these two new apps; they cannot establish reliability for all apps, markets and languages.

### Revised extraction method

The production candidate stage now creates one-to-six-word Unicode source spans within sentence boundaries, retaining numerals, spelled-out quantities, negation and meaningful initial modifiers such as `only`, `too`, `не`, `немає` and `після`. A nearby negator or quantity prevents an affirmative-looking or numberless fragment from being emitted in its scope. Connectors and discourse starts are restricted, and a specific guard removes the misleading Lithuanian `Spotify yra puiki` fragment after `Nepasakyčiau, kad`. A limited positive-word guard avoids treating isolated praise as a complaint in mixed or ironic reviews. This is a conservative heuristic, not full syntactic parsing; it can still miss dialect, distant scope and implicit complaints. Candidates remain exact substrings of the normalized title/body analysis copy, with a maximum of 120 balanced across the review.

All four approaches receive these same candidates. The embedding methods rank model cosine similarity with a small specificity/qualifier bonus, clause diversity penalty and lexical overlap suppression. TF-IDF uses one-to-six-word features with the same candidate selection. Deduplication keeps different numerals and negations distinct; it suppresses nested same-meaning variants but can retain some longer/shorter variants if the longer one adds a material qualifier. Each returned keyword includes `text`, raw model `score`, a separate surrounding-clause `evidence_span`, and start/end offsets in the normalized analysis text. The original review text is not changed. Ratings, gold annotations and arena files remain outside the production extractor. The pipeline still uses Cardiff to decide which reviews are negative, and language-specific common-keyword aggregation still counts unique review IDs.

The old Spotify cases were used **only to debug and measure regression**. Candidate tests now require `too much ads`, `only 6 skips`, `2 minutes of ads`, `3 times a day`, colloquial Ukrainian `три реклами` and `після двох пісень`, and negated login failures before ranking. They also reject cross-sentence and broken fragments, preserve distinct numeric restrictions in deduplication, and verify that evidence and offsets point into the normalized source. Those cases are not presented as independent proof of generalization. Final model comparison uses the held-out set above, which was frozen before its predictions.

### Independent held-out scores

Every method produced 195/195 results without an inference error. The same conservative one-to-one normalized phrase matcher and macro per-review Precision@5, Recall@5 and F1@5 as v1 are used, so wording or inflection differences can score as misses even when semantically valid. Cells are **P / R / F1**. Ukrainian and English are separate primary slices; `Other` combines 18 Polish and 20 Russian reviews.

| Method | All 195 | Ukrainian 77 | English 80 | Other 38 |
| --- | --- | --- | --- | --- |
| TF-IDF | .151 / .300 / **.193** | .189 / .360 / **.240** | .139 / .273 / **.177** | .096 / .233 / **.133** |
| MiniLM | .177 / .370 / **.229** | .217 / .425 / **.276** | .151 / .305 / **.194** | .149 / .393 / **.207** |
| E5-small | .197 / .403 / **.254** | .252 / .484 / **.320** | .177 / .364 / **.229** | .127 / .321 / **.174** |
| BGE-M3 | .194 / .414 / **.253** | .250 / .510 / **.324** | .170 / .355 / **.221** | .129 / .345 / **.180** |

E5-small has the highest overall and English F1. BGE-M3 leads Ukrainian F1 by **.0034**, a very small difference; MiniLM leads the Polish/Russian combined slice, whose support is limited. A paired 5,000-draw bootstrap over the 195 reviews gives a 95% percentile interval of **+.0027 to +.0483** for E5 minus MiniLM F1, and **−.0226 to +.0210** for BGE minus E5. These intervals describe sampling uncertainty within this particular enriched dataset; they do not account for annotation uncertainty or distribution shift. The 165 reviews with unflagged keyword labels retain the pattern: TF-IDF .197, MiniLM .239, E5 .268, BGE .262 F1. By app, E5 scores .239 on 90 Duolingo and .267 on 105 Telegram reviews; BGE scores .227 and .275 respectively. The between-app pattern is a limitation of this two-app test.

### Held-out CPU cost

The same Windows CPU host, local checkpoint cache, model revisions and 120-candidate limit were used. These are single-run wall-clock measurements after load, not a controlled hardware benchmark. TF-IDF time includes corpus fitting.

| Method | Load or fit (s) | Mean inference (s/review) | Total (s) | RSS after run (MB) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.0 | .008 | 1.5 | 125 |
| MiniLM | 8.2 | .512 | 108.1 | 845 |
| E5-small | 10.6 | .619 | 131.4 | 808 |
| BGE-M3 | 8.5 | 6.871 | 1348.2 | 1553 |

BGE-M3 was about **11.1 times** slower than E5-small per review, without a demonstrated overall F1 advantage on this set. The next semantic audit checks whether its phrases nevertheless carry meaning better. The full predictions, per-method metrics, paired comparison and source offsets are in `results_heldout_v1/`; all 195 output IDs, error fields and source slices were validated.

### Independent blind semantic audit

Forty held-out reviews were selected with seed `20260924` **before final predictions**: ten each in Ukrainian, English, Polish and Russian. For every review the four result lists were shuffled under A–D, with model scores, identities, gold labels and ratings hidden. Two separate evaluators each judged 20 language-balanced reviews, five per language. They read title, body, candidate phrase and its evidence clause, then marked useful complaint phrases, misleading phrases, negation/number losses, generic or duplicated phrases, and major missed complaints. The evaluator outputs were unblinded only after both files were complete. The anonymous inputs, mapping, judgments and aggregation script are saved under `datasets/heldout_crossapp_v1/` and `results_heldout_v1/`. This is an **LLM blind audit**, not qualified human adjudication; the two evaluators saw different halves, so small utility differences are especially uncertain.

| Method | Utility / 5 | Useful phrases / all | Meaning errors | Negation or number losses | Generic or duplicate | Major complaints missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 2.050 | .253 (45/178) | 17 | 5 | 124 | 47 |
| MiniLM | 2.825 | .404 (69/171) | 12 | 3 | 104 | 34 |
| E5-small | 2.875 | .407 (70/172) | 14 | 3 | 103 | 30 |
| BGE-M3 | 2.975 | .413 (71/172) | 14 | 4 | 101 | 32 |

BGE has a small +0.10 mean utility and +0.006 useful-phrase-share advantage over E5 in this sample, but the same 14 meaning errors, one extra negation/number loss and two more missed major complaints. E5 has the highest Ukrainian (3.8/5) and English (2.7/5) utility; BGE has the highest Polish (2.7/5), while MiniLM leads Russian (3.1/5, tied with BGE). Each language cell contains only ten reviews. The methods still emit many incomplete, generic or overlapping snippets: **roughly 40%** of the neural outputs were judged individually useful. None is a reliable automatic issue counter without evidence inspection or human review.

Concrete held-out failures include a Polish registration complaint where `4,99 zł` was truncated to `4`, changing the claimed price; a Ukrainian complaint that chat taps do nothing alongside working settings, where `налаштування працюють` was extracted as if it were the issue; and an English subscription complaint where willingness to pay once was treated as a complaint. These failures are in the saved judgments and show why an exact source substring alone is insufficient. The decimal-comma and distant-context cases remain unresolved in this frozen v2 method, rather than being hidden by a broad performance claim.

### Final production integration after the v2 comparison

The historical v1 BGE-M3 selection above is superseded for production. `KeywordExtractor` now defaults to **`intfloat/multilingual-e5-small`**, exact revision **`614241f622f53c4eeff9890bdc4f31cfecc418b3`**, with the benchmark's `query: ` prefix for both the review text and every candidate. The v2 candidate limit is 120, ranking uses cosine similarity plus the recorded specificity/qualifier/diversity rules, and up to five source phrases are returned. The API, per-review error handling, batched candidate encoding, lazy model reuse, evidence spans and offsets, and per-language aggregation by unique review ID remain in production. Cardiff sentiment is unchanged. No production import reads Model Arena data or results. The other three arena methods and their recorded outputs remain available.

Local regression fixes preserve decimal-comma and decimal-point quantities with units, including `4,99 zł`, `2.5 GB`, `8,99 eurus`, and version `12.7`; prevent a selected price or size phrase from ending just before its unit; keep negation and limits such as `not working`, `Only 4 skips`, and `працюють через раз`; and suppress plainly affirmative clauses such as `налаштування працюють` and `bots make it somewhat bearable` when they occur in negative reviews. A clause-end correction also makes `evidence_span` cover the full surrounding clause. These are candidate guards, **not** a phrase-level sentiment classifier. Extracted phrases are evidence-backed **keyphrases**, not individually labeled complaints or confirmed product issues. Downstream issue metrics must inspect the original text and evidence before assigning a problem category.

The final patch changes candidate lists for **19/255** old Spotify and **13/195** held-out negative reviews. Corrected evidence clauses also affect **128/255** and **92/195** reviews respectively, and can alter ranking. All **1,800** saved v2 predictions across four methods and both sets were rechecked for IDs, source offsets, evidence containment and absent errors, but checkpoints were **not rerun** on those sets. Thus the published v2 F1 and semantic-audit numbers above describe the **pre-patch** extraction method and must not be presented as validated final production metrics. This recheck is not a new independent test. Focused regression tests and the full automated suite passed **44/44**. Real pinned-E5 regression inference on the known Polish price case now returns `aplikacja wymaga zakupu subskrypcji 4,99 zł`; the Ukrainian intermittent-operation phrase `працюють через раз` remains, while `налаштування працюють` and the English `bearable` clause are absent. See `results/production_regression_e5_final.json`. A separate real offline checkpoint smoke on four previously collected Duolingo reviews outside the frozen evaluation sets (two Ukrainian, two English) ran Cardiff → negative review selection → pinned E5 → language aggregation: all four were negative, each produced five phrases without inference errors, source offsets/evidence and unique-ID aggregation validated, and both loaded model instances were reused. See `results/production_smoke_e5_final.json`.

Remaining limits: the heuristics cannot reliably resolve distant negation, irony, praise mixed with complaints, or all language-specific number formats. The price case is fixed, but nearby affirmative context can still yield a fragment such as `дзвінки`; this output is a keyphrase and must not be counted as a complaint on its own. The four-review smoke checks execution and structure, not extraction quality; the held-out set has two apps, enriched negative ratings and non-adjudicated labels. The final production F1 after the local patch has **not** been independently measured. Cardiff's model card does not confirm commercial usage rights, as documented in the sentiment report.
