# corpus-gaps

This repository asks one question: **does a retrieval system know what its corpus does not contain?** It contains a small research corpus pipeline (`corpus_brain.py`) and a study that measures how well common retrieval scores can tell a covered question from an uncovered one.

The question matters for any system that answers from documents. If a question is not covered, the system should say so rather than answer from the nearest text it can find. [qualm](https://github.com/Dankaro-projects/qualm) asks the same question of the language model's own confidence; this study asks it of the retrieval step that comes before.

## Summary

- **Unrelated questions are easy.** For questions on subjects the corpus never touches, every signal tested separates them perfectly from covered questions (AUROC 1.000).
- **A missing topic is detectable.** When every document on a topic is removed, all three signals reach an AUROC of about 0.94 to 0.95. Keyword search (BM25) does as well as either embedding model.
- **A missing document is hard.** When only the document that answers the question is removed and related documents remain, the best signal reaches an AUROC of 0.80, and the best threshold, fitted on half the questions, classifies 74% of the other half correctly. This is the case that matters most in practice, because a related document is exactly what a system is tempted to answer from.
- **The threshold belongs to the corpus and the model.** The learned threshold is about 0.77 for `bge-small-en-v1.5` and about 0.59 for `all-MiniLM-L6-v2`. The library's previous default of 0.72, measured on an earlier corpus, classifies missing documents no better than chance here (56%), and is meaningless for the second model.
- **Filtering reference lists hardly matters for retrieval.** About 18% of chunks are reference lists. Removing them changes no detection result beyond its confidence interval, and it improves the match between unsupervised clusters and the true topics only slightly.

## Second study: why a missing document is hard to detect

A pre-registered follow up tested an explanation on four public benchmarks (SciFact, NFCorpus, FiQA and SCIDOCS) with two embedding models: when a missing document has close neighbours in the corpus, they score almost as high and hide the gap. The plan was committed before any data was loaded (`docs/PREREGISTRATION.md`), and the results are reported against it (`docs/REDUNDANCY_STUDY.md`).

- Of four pre-registered hypotheses, one failed (redundancy did not predict the drop in the best score across all combinations), two were supported (detection is harder where the corpus is more redundant, by an AUROC of 0.05; redundancy adds predictive information in 6 of 8 combinations), and the geometric bound held with no violation, though loosely.
- On these benchmarks the best score barely reveals a missing document: detection AUROC is 0.55 to 0.69.
- An exploratory analysis, not pre-registered, points to the reason: the best match is a relevant document for only 23% to 59% of queries, and where it is not, removing the relevant documents cannot change the best score. Among the queries where retrieval works, redundancy does predict how hidden the gap is. This is the hypothesis for the next pre-registered test.

## The corpus

The corpus is built from 60 public URLs on six research topics: analytic standards in intelligence work, expert elicitation, forecasting calibration, evaluation of language models, debate between several models, and research agents. The list is in `seeds.csv`.

| | Count |
|---|---|
| URLs seeded | 60 |
| Fetched with usable text | 42 (70%) |
| Refused by the server (HTTP 403 or 406) | 12 |
| Returned too little text | 4 |
| PDF could not be parsed | 2 |
| Chunks | 2,104 |
| Chunks flagged as reference lists | 372 (17.7%) |

One page counted as fetched is a library login page, not the document. The fetcher accepts any page with more than 400 characters of text, so it cannot tell a paywall from content. That page stays in the corpus, because real corpora contain such pages, but it is not used as a test target.

## Method

Each of 39 questions was written by hand, in different words from its source, to be answered by one specific document. Twenty further questions cover subjects the corpus never touches. The questions are in `experiments/questions.jsonl`.

Each document question is searched in three conditions, and the best match score is recorded:

| Condition | What is removed from the corpus | What it tests |
|---|---|---|
| Present | Nothing | The score of a covered question |
| Document absent | The target document, and any copy of it | Whether the system notices that the one source it needs is missing, while related documents remain |
| Topic absent | Every document on the target's topic | Whether the system notices that a whole topic is missing |

The unrelated questions are searched against the full corpus. Because the conditions are built by removing documents, the true answer to "is this covered?" is known in every case, without anyone judging coverage by eye. Two pairs of documents are the same work published in two places; when one is removed, so is its twin.

Four signals are compared: the best cosine similarity with `BAAI/bge-small-en-v1.5`, the best cosine similarity with `sentence-transformers/all-MiniLM-L6-v2`, the best BM25 score from SQLite full text search, and a random number as the floor that any signal must beat. The hybrid search in the library fuses rankings rather than scores, so it has no absolute score and is measured only for retrieval.

AUROC is the probability that a covered question scores higher than an uncovered one; 0.5 is chance and 1.0 is perfect separation. Intervals are 95% bootstrap intervals over questions (2,000 resamples). For thresholds, the accuracy maximising threshold is fitted on a random half of the questions and tested on the other half, 1,000 times.

## Results

### Can the score tell covered from uncovered?

AUROC with 95% intervals, all chunks included:

| Signal | Present against document absent | Present against topic absent | Present against unrelated |
|---|---|---|---|
| Vector, bge-small | **0.799** [0.721, 0.878] | 0.943 [0.890, 0.986] | 1.000 |
| Vector, MiniLM | 0.705 [0.634, 0.788] | **0.954** [0.908, 0.988] | 1.000 |
| Keyword, BM25 | 0.739 [0.667, 0.819] | 0.948 [0.897, 0.990] | 1.000 |
| Random | 0.613 [0.462, 0.761] | 0.446 [0.339, 0.560] | 0.462 [0.300, 0.624] |

The ranges of the scores explain the pattern. With bge-small, covered questions scored between 0.682 and 0.879, questions whose document was removed between 0.682 and 0.831, and unrelated questions between 0.488 and 0.636. The first two ranges overlap almost completely, so no threshold can separate them cleanly.

### Does a threshold learned on some questions work on others?

| Signal | Condition | Held out accuracy | Learned threshold (median, 5th to 95th percentile) |
|---|---|---|---|
| Vector, bge-small | Document absent | 0.738 [0.650, 0.825] | 0.766 (0.764 to 0.773) |
| Vector, bge-small | Topic absent | 0.852 [0.775, 0.925] | 0.764 (0.737 to 0.773) |
| Vector, MiniLM | Document absent | 0.605 [0.550, 0.650] | 0.599 (0.548 to 0.684) |
| Vector, MiniLM | Topic absent | 0.847 [0.750, 0.900] | 0.586 (0.548 to 0.616) |
| Keyword, BM25 | Document absent | 0.638 [0.575, 0.700] | 21.8 (16.6 to 25.6) |
| Keyword, BM25 | Topic absent | 0.888 [0.800, 0.950] | 17.8 (16.6 to 19.3) |

The fixed threshold of 0.72, the library's previous default, gives these accuracies:

| Model | Document absent | Topic absent | Unrelated |
|---|---|---|---|
| bge-small | 0.564 | 0.769 | 0.983 |
| MiniLM | 0.538 | 0.564 | 0.424 |

For bge-small, the learned threshold is stable across splits, but it does not transfer from the earlier corpus. For MiniLM, whose scores are on a different scale, 0.72 is below chance on unrelated questions, because it labels most covered questions as gaps.

### Does retrieval find the right document when it is there?

The target document in the top 5 results, out of 39 questions:

| Method | All chunks | Reference lists removed |
|---|---|---|
| Vector, bge-small | 37 | 38 |
| Vector, MiniLM | 39 | 37 |
| Keyword, BM25 | 36 | 36 |
| Hybrid (BM25 and bge-small fused by reciprocal rank) | **39** | **39** |

Finding a document that is present is not the difficulty. Knowing when it is absent is.

### Do reference lists matter?

A chunk is flagged as a reference list when it carries at least 4 citation markers per 1,000 characters, such as "et al.", "arXiv", a DOI, a year in brackets or a URL. On a hand checked sample, 13 of 15 flagged chunks were reference lists and 2 were body text dense with citations. Some author lists without such markers are missed.

| Measure | All chunks | Reference lists removed |
|---|---|---|
| AUROC, document absent, bge-small | 0.799 | 0.800 |
| AUROC, document absent, MiniLM | 0.705 | 0.718 |
| AUROC, document absent, BM25 | 0.739 | 0.750 |
| Clusters out of 14 made mostly of references (5 seeds) | 1 in every seed | not applicable |
| Adjusted mutual information between 14 clusters and the 6 true topics (mean of 5 seeds) | 0.518 | 0.530 |

Removing reference lists removes one junk cluster and nudges every measure in the right direction, but no change is larger than its uncertainty. It is worth doing for topic discovery and not a priority for retrieval.

## What this study does not show

- **Scale.** The corpus has 42 documents and the study 59 questions. The intervals are wide, and the numbers describe this corpus. The method, removing documents to create known gaps, transfers; the thresholds do not, which is the point.
- **Answer quality.** The library can answer questions with Claude from the retrieved excerpts and audit the citations, but no API key was available, so that step has not been run. Whether a language model declines correctly when retrieval scores are low is the natural next measurement.
- **Better signals.** Only the single best match score was tested. The gap between the first and second match, agreement between keyword and vector rankings, or a reranking model might separate the hard case better.
- **Harvesting.** The search engine harvester in the library has never been run.

## Repository layout

| Path | Contents |
|---|---|
| `corpus_brain.py` | The pipeline: seed, fetch, chunk, embed, keyword and vector search, fusion, clustering, gap report, threshold calibration, grounded answering prompt and citation audit |
| `test_brain.py` | 14 checks of the paths that need no network |
| `seeds.csv` | The 60 public URLs and their topics |
| `experiments/build_corpus.py` | Builds the corpus and embeds it with both models |
| `experiments/questions.jsonl` | The 59 questions and their target documents |
| `experiments/run_gap_study.py` | The study: conditions, AUROC, threshold transfer, retrieval and clustering |
| `results/summary.json` | Every number in this README |
| `results/scores.jsonl` | The score of every question in every condition |
| `results/reference_rule_sample.json` | The sample used to check the reference rule by hand |
| `docs/PREREGISTRATION.md` | The plan of the second study, committed before any data was loaded |
| `docs/REDUNDANCY_STUDY.md` | The results of the second study against its plan |
| `experiments/redundancy_study.py`, `experiments/redundancy_exploratory.py` | The second study and its labelled exploratory follow up |
| `results/redundancy/` | Every per query value of the second study |

## Running it

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run python test_brain.py                                                     # no network needed
uv run --extra fetch --extra embed python experiments/build_corpus.py           # about 4 minutes
uv run --extra embed --extra analysis python experiments/run_gap_study.py       # under a minute
```

Pages change and servers refuse requests, so a new build will not fetch exactly the same text. The database is not committed; `results/` records the run reported here, made on 18 September 2026.

## Changes to the library

The study found and fixed one defect. The embeddings table used the chunk identifier alone as its key, so a chunk could hold only one model's vector, and embedding with a second model silently did nothing. The key is now the chunk and the model together, and the embedding step fills in each model separately. The note on the gap threshold was updated with the measurements above.

## Licence

MIT. See `LICENSE`.
