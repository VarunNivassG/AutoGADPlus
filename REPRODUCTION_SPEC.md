# Reproduction Specification — AutoGAD Phase 1

## 1. Problem setting

Transductive, unsupervised node anomaly detection on a static attributed graph.

- Graph: `G = (V, E, X)`.
- Detector output: one anomaly score `s_i` per node.
- Larger anomaly score means more anomalous.
- Ground-truth anomaly labels are **not available to AutoGAD model selection**.
- Ground-truth labels are used only for final evaluation (e.g. ROC-AUC).

## 2. AutoGAD decomposition

AutoGAD has two logically independent parts:

1. Internal evaluation strategy: Contrast Score Margin (CSM).
2. Search method: discretization + grid search.

For every candidate configuration `lambda_m`:

```text
run base detector f(lambda_m; G)
        ↓
score vector s_m(G)
        ↓
CSM(s_m(G))
        ↓
store result
```

Select the configuration with maximum CSM.

## 3. CSM used by the paper

The AutoGAD paper modifies the original Contrast Score Margin to avoid relying on a second pseudo-normal top-k set.

Let:

- `O_hat`: top-k predicted anomalies.
- `I_tilde`: remaining `n-k` nodes.
- `mu_O`, `var_O`: mean and variance of scores in `O_hat`.
- `mu_I`, `var_I`: mean and variance of scores in `I_tilde`.

Then:

```text
CSM = (mu_O - mu_I) / sqrt(var_O + var_I)
```

No true anomaly labels are used in this calculation.

## 4. ANEMONE reproduction boundary

Native ANEMONE follows the published formulation:

1. For each target node, generate two random-walk-with-restart ego-nets of fixed size `K`.
2. Put the target node first in each local node ordering.
3. Mask the target node attributes to zeros before GNN processing.
4. Patch-level contrast:
   - GNN output of masked target node: `h_p_i`.
   - Original target features projected to the same space: `z_p_i`.
   - Positive score: bilinear(`h_p_i`, `z_p_i`).
   - Negative score: bilinear(`h_p_j`, `z_p_i`) for another target `j != i` from the same mini-batch.
5. Context-level contrast:
   - GNN produces node embeddings for the target's ego-net.
   - Average pooling gives context embedding `h_c_i`.
   - Original target features projected into the context space: `z_c_i`.
   - Positive/negative scores are formed analogously.
6. Joint loss:

```text
L = alpha * L_c + (1 - alpha) * L_p
```

7. Inference:
   - Generate `R` ego-nets for the patch and context estimators.
   - Sample an equal number of negative nodes.
   - For each view and round:

```text
b_view,j = negative_score_view,j - positive_score_view,j
```

   - Per-view statistical score:

```text
y_view = mean(b_view) + population_std(b_view)
```

   - Final score:

```text
y_i = alpha * y_c_i + (1 - alpha) * y_p_i
```

## 5. Paper ANEMONE search space in AutoGAD

```text
K     = {2,3,4,5}
alpha = {0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1}
```

## 6. Dataset protocol

Injected anomaly datasets use the structural/contextual anomaly construction followed by the ANEMONE/CoLA family. The included injector reproduces the public CoLA implementation's main logic.

## 7. Evaluation protocol

For each experiment:

- run five independent seeds for the full-paper setting;
- record raw anomaly scores;
- calculate ROC-AUC only after model selection;
- record CSM, selected configuration, runtime, and AUC.

## 8. Reproduction integrity rules

Never:

- use anomaly labels to pick `K`, `alpha`, or another SSL hyperparameter;
- compute CSM from ROC-AUC;
- replace CSM with ground-truth metrics during selection;
- silently change anomaly injection while claiming exact reproduction.

The official author repository remains the reference for legacy implementation details and paper-table reproduction.


## ANEMONE negative sampling correction

The ANEMONE configuration keeps separate training negative-sampling rounds for the patch and context discriminators (`negsamp_round_patch` and `negsamp_round_context`). This matches the official ANEMONE model structure, where the discriminators emit one positive score followed by the requested number of cyclic same-batch negative scores. The training loss averages the positive and negative terms over all configured negative rounds.
