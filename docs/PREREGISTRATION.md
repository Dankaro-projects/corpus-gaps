# Pre-registration: why a missing document is hard to detect

Written and committed on 18 September 2026, before any of the datasets below was downloaded or embedded. The commit that adds this file is the timestamp. Any change after the first run of the analysis will be recorded in a dated section at the end, and results will be reported against the plan as written here.

## Where the idea comes from

The first study in this repository (see the README) found that a retrieval score separates covered from uncovered questions well when a whole topic is missing (AUROC about 0.95), but poorly when only the one document that answers a question is missing and related documents remain (AUROC 0.80 at best). That study was exploratory, and the explanation below was formed after seeing its results. The datasets named below have not been used in this repository, so they are the confirmatory test.

## The explanation

When the document that answers a question is removed, the best remaining score comes from some other document. If the removed document has a close neighbour in the corpus, that neighbour will usually score almost as high as the removed document did, and the gap in coverage is hidden.

For unit vectors this can be stated exactly. For a query q, a relevant document t and any other document n, the angle between q and n is at most the sum of the angles between q and t and between t and n. It follows that

    cos(q, n) ≥ cos(q, t)·cos(t, n) − sqrt((1 − cos²(q, t))·(1 − cos²(t, n)))

So the score left after removal is bounded below by a quantity that depends only on how well the removed document matched the query and on how similar the removed document is to its nearest remaining neighbour. The second quantity, which this plan calls **redundancy**, can be computed from the corpus alone, before any question is asked.

The explanation predicts that the harder a gap is to detect, the more redundant the corpus is around the missing document.

## Data

Four datasets from the BEIR benchmark, using their test queries and relevance judgements: **SciFact, NFCorpus, FiQA-2018 and SCIDOCS**. ArguAna is excluded in advance, because its relevant document is a counter argument to the query, which is a different relation from answering it. The corpus of the first study is analysed with the same code, and reported separately as exploratory.

Two embedding models: `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`. Documents are embedded as title plus text, truncated to the model's maximum length. Queries use the query prefix the model expects, as in the library. This gives 8 confirmatory combinations of dataset and model.

## Definitions

For each query q with relevant set T (documents judged relevant with a score above 0) in corpus C:

- **Present score** s_p: the highest cosine similarity between q and any document in C.
- **Target score** s_t: the highest cosine similarity between q and any document in T.
- **Absent score** s_a: the highest cosine similarity between q and any document in C with every member of T removed.
- **Margin** Δ = s_p − s_a.
- **Redundancy** r: the highest cosine similarity between any member of T and any document outside T. It does not use the query.
- **Bound** b: the right hand side of the inequality above, with cos(q, t) = s_t and cos(t, n) = r.

Detection is scored as in the first study: the AUROC of s_p against s_a over all queries, where s_p is treated as covered and s_a as uncovered, with a 95% bootstrap interval over queries (2,000 resamples, paired by query).

## Hypotheses and how each could fail

**H1. Redundancy predicts how much the score drops.** In each of the 8 combinations, the Spearman correlation between r and Δ is negative. The hypothesis fails if the 95% bootstrap interval of the correlation includes zero or lies above it in 3 or more of the 8 combinations.

**H2. Redundancy predicts detection difficulty.** Queries are split into thirds by r within each combination. The detection AUROC in the most redundant third is lower than in the least redundant third. The hypothesis fails if, pooling the 8 combinations, the 95% interval of the difference (least redundant minus most redundant) includes zero or lies below it.

**H3. Redundancy adds information beyond the target score.** In each combination, a linear regression of s_a on s_t and r explains more variance than a regression on s_t alone. The hypothesis fails if the 95% bootstrap interval of the increase in R² includes zero in 3 or more of the 8 combinations.

**H4. The bound holds.** Every query satisfies s_a ≥ b, allowing for floating point error of 1e-5. This is a check of the mathematics and of the code, not a finding. Any violation means an error in the code.

## Also reported, with no hypothesis

- Detection AUROC for each combination, with its interval.
- How tight the bound is: the distribution of s_a − b.
- Whether keyword search (BM25) shows the same dependence on redundancy, measured with a redundancy defined on BM25 document similarity if that proves practical. If not, this is reported as not done.

## What would change the plan

If a dataset cannot be downloaded or embedded within reasonable time on a laptop CPU, it is dropped, the drop is reported, and the failure thresholds scale with the number of remaining combinations (3 of 8 becomes the same proportion of the remainder, rounded up). No dataset will be dropped because of its results.
