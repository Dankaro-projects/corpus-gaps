# Results of the confirmatory test

The plan is in `docs/PREREGISTRATION_2.md`, committed on 18 September 2026 at 16:40, before any of the datasets below was downloaded. It tests, on new data, a finding that the first study reached by exploratory analysis. No amendment was made, and no dataset was dropped.

## Data and run

Five datasets from the BEIR benchmark that the earlier studies did not use, each embedded with two models, gave 10 combinations. All 10 had at least 20 answered queries, so all 10 count, and the pre-registered failure threshold is 4 combinations.

| Dataset | Documents | Test queries |
|---|---|---|
| TREC-COVID | 171,332 | 50 |
| CQADupStack, android | 22,998 | 699 |
| CQADupStack, english | 40,221 | 1,570 |
| CQADupStack, gaming | 45,301 | 1,595 |
| CQADupStack, physics | 38,316 | 1,039 |

Code: `experiments/confirmatory_study.py`. Every per query value is in `results/confirmatory/`.

## Verdicts against the plan

| Hypothesis | Failure rule | Result | Verdict |
|---|---|---|---|
| H5. Among answered queries, redundancy is negatively correlated with the drop in the best score | Fails if the interval includes zero or lies above it in 4 or more of 10 combinations | Supported in 8 of 10 | **Supported** |
| H6. Among answered queries, detection is harder in the most redundant third than in the least redundant third | Fails if the pooled interval of the difference includes zero or lies below it | Difference 0.090 [0.064, 0.109] | **Supported** |
| H7. The pooled correlation is at least as strong as −0.15 | Fails if the mean correlation is above −0.15 | Mean −0.218 | **Supported** |

## Detail by combination

An answered query is one whose best match in the full corpus is a relevant document.

| Dataset and model | Answered queries | Detection AUROC, all queries | Detection AUROC, answered | H5: Spearman among answered | H6: AUROC, least and most redundant third |
|---|---|---|---|---|---|
| TREC-COVID, bge-small | 45 of 50 (90%) | 0.740 | 0.774 | 0.119 [−0.191, 0.419] | 0.725 and 0.719 |
| TREC-COVID, MiniLM | 34 of 50 (68%) | 0.655 | 0.710 | −0.202 [−0.559, 0.189] | 0.751 and 0.711 |
| android, bge-small | 255 of 699 (36.5%) | 0.603 | 0.782 | **−0.185** [−0.309, −0.060] | 0.837 and 0.738 |
| android, MiniLM | 297 of 699 (42.5%) | 0.624 | 0.786 | **−0.273** [−0.377, −0.168] | 0.856 and 0.744 |
| english, bge-small | 582 of 1,570 (37.1%) | 0.593 | 0.743 | **−0.324** [−0.395, −0.243] | 0.829 and 0.700 |
| english, MiniLM | 629 of 1,570 (40.1%) | 0.616 | 0.783 | **−0.222** [−0.294, −0.143] | 0.858 and 0.736 |
| gaming, bge-small | 714 of 1,595 (44.8%) | 0.615 | 0.752 | **−0.261** [−0.329, −0.190] | 0.812 and 0.719 |
| gaming, MiniLM | 716 of 1,595 (44.9%) | 0.638 | 0.800 | **−0.232** [−0.301, −0.158] | 0.866 and 0.780 |
| physics, bge-small | 368 of 1,039 (35.4%) | 0.605 | 0.784 | **−0.352** [−0.445, −0.264] | 0.844 and 0.729 |
| physics, MiniLM | 376 of 1,039 (36.2%) | 0.608 | 0.778 | **−0.251** [−0.352, −0.148] | 0.840 and 0.741 |

Values in bold meet the pre-registered criterion. Intervals are 95% bootstrap intervals over queries. The two combinations that do not meet it are both TREC-COVID, which has 50 queries; the plan anticipated wide intervals there.

As expected by construction, detection AUROC for queries that are not answered is exactly 0.5 in every combination: their best score does not change when the relevant documents are removed.

## Conclusion across the three studies

1. **A retrieval score can only reveal a missing document if retrieval found that document in the first place.** On the nine public benchmarks used across the two pre-registered studies, the best match was a relevant document for between 23% and 90% of queries, and for most datasets fewer than half. For every other query, the best score gives no evidence of the gap at all.
2. **Among the queries that retrieval answers, redundancy hides the gap.** The more similar the missing document is to a document that remains, the smaller the drop in the best score and the harder the gap is to detect. This was found by exploration on four datasets and confirmed on five new ones, with a mean correlation of −0.22 and a difference in detection AUROC of 0.09 between the least and most redundant thirds of queries.
3. **Redundancy is measurable before any question is asked.** It depends only on the corpus. A corpus with many near duplicates is one where missing sources will be hard to notice, and that can be assessed in advance.

## Limits

- Only two small embedding models were used, and only the single best cosine score as the detection signal. Larger models, rerankers or other signals may behave differently.
- The confirmatory datasets are dominated by one family, the CQADupStack forums, whose relevant documents are duplicate questions. TREC-COVID, the only other new dataset, has too few queries to settle the question on its own.
- The effect is moderate. Redundancy explains part of the variation in how hidden a gap is, not most of it.
