# Results of the pre-registered redundancy study

The plan is in `docs/PREREGISTRATION.md`, committed on 18 September 2026 at 16:06, with one amendment to the definition of the bound committed 21 seconds later. Both commits came before any data was downloaded. This report follows the plan. Analyses that were not in the plan are in a separate section and are labelled exploratory.

## Data and run

Four BEIR datasets and two embedding models gave the 8 confirmatory combinations. No dataset was dropped. The first study's corpus was analysed with the same code as an exploratory ninth dataset.

| Dataset | Documents | Test queries | Median redundancy, bge-small | Median redundancy, MiniLM |
|---|---|---|---|---|
| SciFact | 5,183 | 300 | 0.849 | 0.672 |
| NFCorpus | 3,633 | 323 | 0.943 | 0.870 |
| FiQA-2018 | 57,638 | 648 | 0.870 | 0.700 |
| SCIDOCS | 25,657 | 1,000 | 0.908 | 0.786 |
| First study corpus (exploratory) | 2,104 chunks | 39 | 0.905 | 0.802 |

Code: `experiments/redundancy_study.py`. Every per query value is in `results/redundancy/`.

## Verdicts against the plan

| Hypothesis | Failure rule | Result | Verdict |
|---|---|---|---|
| H1. Redundancy predicts how much the best score drops (Spearman correlation below zero) | Fails if not supported in 3 or more of 8 combinations | Supported in 3 of 8 | **Not supported** |
| H2. Detection is harder in the most redundant third of queries than in the least redundant third | Fails if the pooled 95% interval of the difference includes zero | Difference 0.050 [0.035, 0.064] | **Supported, with a small effect** |
| H3. Redundancy adds explained variance beyond the target score when predicting the score left after removal | Fails if not supported in 3 or more of 8 combinations | Supported in 6 of 8 | **Supported** |
| H4. The score left after removal is never below the geometric bound | Any violation means an error in the code | 0 violations in 4,529 checked queries; 13 excluded because a cosine was not positive | **Holds** |

### Detail by combination

| Dataset and model | Detection AUROC | H1: Spearman (redundancy, drop) | H2: AUROC, least and most redundant third | H3: gain in R² |
|---|---|---|---|---|
| SciFact, bge-small | 0.686 [0.660, 0.713] | **−0.289** [−0.391, −0.174] | 0.769 and 0.611 | **0.207** [0.133, 0.289] |
| SciFact, MiniLM | 0.646 [0.622, 0.671] | **−0.306** [−0.407, −0.193] | 0.732 and 0.586 | **0.223** [0.136, 0.317] |
| NFCorpus, bge-small | 0.593 [0.575, 0.611] | 0.070 [−0.035, 0.176] | 0.578 and 0.597 | 0.008 [0.000, 0.026] |
| NFCorpus, MiniLM | 0.587 [0.570, 0.606] | 0.074 [−0.033, 0.180] | 0.578 and 0.597 | 0.003 [0.000, 0.017] |
| FiQA, bge-small | 0.604 [0.591, 0.619] | −0.046 [−0.120, 0.030] | 0.635 and 0.578 | **0.062** [0.033, 0.098] |
| FiQA, MiniLM | 0.603 [0.588, 0.619] | 0.005 [−0.071, 0.080] | 0.616 and 0.591 | **0.074** [0.036, 0.120] |
| SCIDOCS, bge-small | 0.549 [0.541, 0.558] | **−0.105** [−0.167, −0.040] | 0.572 and 0.536 | **0.024** [0.011, 0.044] |
| SCIDOCS, MiniLM | 0.554 [0.546, 0.563] | −0.041 [−0.107, 0.025] | 0.566 and 0.546 | **0.010** [0.002, 0.024] |

Values in bold meet the pre-registered criterion. Intervals are 95% bootstrap intervals over queries.

### The bound

The bound holds, as mathematics requires, but it is loose. The median gap between the score left after removal and the bound ranges from 0.36 to 0.82 across the combinations. The bound explains why redundancy can matter, but it is too weak to predict the score of a particular query.

## The finding that matters most

**On these public benchmarks, the best retrieval score hardly reveals that the relevant documents are missing.** Detection AUROC is between 0.549 and 0.686, against 0.799 on the first study's corpus. On three of the four datasets, the median drop in the best score when every relevant document is removed is zero.

## Exploratory analysis, not pre-registered

The zero median drop suggested an explanation that the plan did not anticipate: for many queries, the best match was never a relevant document, so removing the relevant documents cannot change the best score. Code: `experiments/redundancy_exploratory.py`; results: `results/redundancy/exploratory.json`.

| Dataset and model | Queries whose best match is relevant | H1 restricted to those queries |
|---|---|---|
| SciFact, bge-small | 59.0% | −0.425 [−0.547, −0.283] |
| SciFact, MiniLM | 50.3% | −0.426 [−0.552, −0.288] |
| NFCorpus, bge-small | 43.3% | −0.215 [−0.354, −0.065] |
| NFCorpus, MiniLM | 42.1% | −0.202 [−0.367, −0.026] |
| FiQA, bge-small | 40.1% | −0.227 [−0.337, −0.105] |
| FiQA, MiniLM | 34.4% | −0.233 [−0.348, −0.101] |
| SCIDOCS, bge-small | 23.4% | −0.116 [−0.251, 0.018] |
| SCIDOCS, MiniLM | 24.4% | −0.089 [−0.214, 0.036] |
| First study corpus, bge-small | 82.1% | −0.555 [−0.790, −0.231] |
| First study corpus, MiniLM | 71.8% | 0.102 [−0.306, 0.473] |

Two things follow, both to be confirmed on new data before they are claimed.

1. **Detection is limited first by retrieval.** A score can only reveal a missing document if that document was the best match when it was present. Where retrieval ranks an irrelevant document first for most queries, as on SCIDOCS, detecting absence from the best score is close to impossible. The first study's corpus looked easier mainly because its questions were written for specific documents, so the best match was relevant for most of them.
2. **Among the queries where retrieval works, redundancy predicts how hidden the gap is.** Restricted to those queries, the correlation between redundancy and the drop is negative, with an interval below zero, in 6 of the 8 benchmark combinations. The two exceptions are both SCIDOCS.

This restriction was chosen after seeing the results, so it may fit this data better than new data. It is a hypothesis for the next pre-registered test, not a result.

## What this means in practice

A system that answers from documents cannot rely on a low best match score to recognise that its corpus lacks the answer. On realistic benchmarks the signal is weak, and it is weakest exactly where the corpus contains close neighbours of the missing source. Declining to answer needs other evidence: whether the answer can be found word for word in the retrieved text, whether repeated answers agree, or a model that judges support explicitly. [qualm](https://github.com/Dankaro-projects/qualm) measured those checks for the generation step.

## Next confirmatory test

To be pre-registered before any new data is used: on BEIR datasets not used here (for example TREC-COVID, HotpotQA and NQ, or subsets of them), (a) the share of queries whose best match is relevant predicts detection AUROC across datasets, and (b) among those queries, redundancy is negatively correlated with the drop in the best score.
