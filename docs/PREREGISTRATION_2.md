# Pre-registration 2: redundancy among queries that retrieval answers

Written and committed on 18 September 2026, before any of the datasets below was downloaded or embedded. The commit that adds this file is the timestamp. Changes made before the first run are recorded in a dated amendment at the end; results are reported against the plan as written.

## Where the hypotheses come from

The first pre-registered study (`docs/PREREGISTRATION.md`, results in `docs/REDUNDANCY_STUDY.md`) found that on four BEIR datasets the best retrieval score barely reveals a missing document, and that redundancy did not predict the drop in the best score across all queries (H1 failed). An exploratory analysis, chosen after seeing the results, suggested why: for many queries the best match is not a relevant document, and for those queries removing the relevant documents cannot change the best score. Among the queries whose best match is relevant, redundancy was negatively correlated with the drop in 6 of 8 combinations.

This plan tests that exploratory finding on data not used before.

## What is accounting and what is a hypothesis

If the best match for a query is not a relevant document, the best score is the same whether or not the relevant documents are present. That follows from the definitions, so it is not tested; it is reported as a description, as the share of queries whose best match is relevant and the detection AUROC within each group.

The hypotheses concern only the queries whose best match is relevant, called **answered queries** below.

## Data

Datasets from the BEIR benchmark not used in the first study, with their test queries and relevance judgements:

- **TREC-COVID** (about 171,000 documents, 50 queries)
- **CQADupStack**, four forums: **android, english, gaming, physics** (about 23,000 to 45,000 documents each)

HotpotQA and NQ were considered and excluded in advance, because their corpora of several million documents cannot be embedded on a laptop in reasonable time. Sampling their corpora was rejected, because sampling changes redundancy, which is the quantity under test.

The same two models as before: `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`, with the same document and query handling. This gives 10 combinations.

TREC-COVID has only 50 queries and many relevant documents per query, so its intervals will be wide. It is included because it is the only feasible new dataset that is not a forum, and its result counts like any other.

## Definitions

As in the first plan, including its amendment: present score s_p, target score s_t, absent score s_a, drop Δ = s_p − s_a, and redundancy r, the highest cosine similarity between any relevant document and any document outside the relevant set. A query is **answered** when its best match in the full corpus is a relevant document, that is, when Δ > 0.

## Hypotheses and how each could fail

**H5. Among answered queries, redundancy predicts the drop.** In each combination, the Spearman correlation between r and Δ over answered queries is negative. It fails if the 95% bootstrap interval includes zero or lies above it in 4 or more of the 10 combinations (the first plan's rule of 3 in 8, scaled and rounded up). A combination with fewer than 20 answered queries is reported but not counted, and the threshold scales with the counted combinations.

**H6. Among answered queries, detection is harder where redundancy is higher.** Within each combination, answered queries are split into thirds by r, and the detection AUROC of the least redundant third is compared with that of the most redundant third. It fails if, pooling the counted combinations, the 95% bootstrap interval of the mean difference (least minus most redundant) includes zero or lies below it.

**H7. The size of the effect.** Among answered queries, the pooled correlation from H5 (the mean over counted combinations) is at least as strong as −0.15. This is a check that the effect is large enough to matter, not only different from zero. It fails if the mean correlation is above −0.15.

Intervals are 95% bootstrap intervals over queries with 2,000 resamples, as before.

## Also reported, with no hypothesis

- The share of answered queries, and the detection AUROC over all queries, per combination.
- The first plan's H1 to H4 on the new data, for comparison.

## What would change the plan

A dataset that cannot be downloaded or embedded is dropped and reported, and the thresholds scale as stated. No dataset is dropped because of its results.
