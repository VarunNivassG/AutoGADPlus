# AutoGAD Reproduction — Phase 1

A clean, implementation-first reproduction workspace for **Towards Automated Self-Supervised Learning for Truly Unsupervised Graph Anomaly Detection** (Li, Wang, van Leeuwen, 2025).

The repository is organized around the paper's actual execution boundary:

```text
Graph + unlabeled nodes
        │
        ▼
   GAD algorithm
   (first: ANEMONE)
        │
        ▼
 anomaly score s_i for every node
        │
        ▼
       CSM
        │
        ▼
 AutoGAD grid search
        │
        ▼
 selected hyperparameters
        │
        ▼
 final anomaly scores
        │
        └──────────────► ROC-AUC (labels used only for evaluation)
```

## What is implemented here

- Native, dependency-light ANEMONE reimplementation in modern PyTorch.
- RWR ego-net generation, target-node masking, patch/node-node contrast, context/node-subgraph contrast, joint loss, multi-round statistical anomaly estimator.
- Exact AutoGAD CSM definition used by the paper's modified internal evaluation strategy.
- Generic grid-search engine that is independent of ANEMONE.
- Paper search spaces for all 10 evaluated methods.
- CoLA-style structural/contextual anomaly injection utility for the citation/social injection datasets.
- `.mat` loader compatible with the datasets used by the authors' public code.
- Label-free model-selection path and separate external evaluation path.
- Registry/adapters for all ten paper methods. Five methods are exposed through official-repository adapters; the five PyGOD-based methods have a lazy PyGOD adapter.
- Toy end-to-end smoke test that requires no external dataset download.
- Official reproduction references and environment notes in `third_party/`.

## Important reproduction note

The official AutoGAD repository currently uses two legacy environments: Python 3.7.8 + Torch 1.10.2 + DGL 0.4.1 for ANEMONE/CoLA/GRADATE/SL-GAD/Sub-CR, and a newer PyTorch Geometric/PyGOD stack for the remaining methods. This project therefore keeps a **native modern implementation for the core Phase-1 path** while also documenting the authors' official repositories and environments. Exact paper-number reproduction should be cross-checked against those official implementations.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke_test.py
```

Expected smoke-test behavior:

```text
1. build toy graph
2. train ANEMONE
3. produce anomaly scores
4. compute CSM for several (K, alpha) candidates
5. select the highest-CSM configuration
6. report ROC-AUC using synthetic labels
```

## Run a real dataset

The simplest supported path is a `.mat` dataset containing `Network`, `Attributes`, and `Label` fields.

```bash
python -m autogad_reproduction.cli run \
  --dataset data/processed/cora.mat \
  --model anemone \
  --search-space anemone \
  --k 150 \
  --device cpu \
  --output outputs/cora_autogad
```

For a faithful AutoGAD model-selection run, do not pass labels into the search engine. Labels are loaded only by the final evaluation stage.

### Research logging and visualizations

Every run writes a reproducibility-oriented artifact directory. By default, use `--output outputs/cora_autogad`. The directory contains:

```text
log.txt                       # chronological human-readable log
events.jsonl                  # machine-readable event stream
run_metadata.json             # environment, dataset, seed, search space
epoch_metrics.csv             # epoch-level training curves
batch_metrics.csv             # only when --log-batches is passed
inference_round_metrics.csv   # per inference round score statistics
search_results.csv            # every hyperparameter configuration + CSM
search_results.json            # selected configuration summary
final_scores.npy
final_node_scores.csv         # node score, rank, and evaluation label when available
plots/
  search_csm_vs_config.png
  search_runtime_vs_config.png
  csm_heatmap_K_alpha.png
  csm_components.png
  training_loss_total.png
  training_loss_patch.png
  training_loss_context.png
  training_discriminator_scores.png
  inference_round_stability.png
  final_score_rank.png
  final_score_distribution.png
  final_roc_curve.png
```

Use `--log-batches` when you need batch-level training traces. The default research log records every search configuration, every epoch, inference-round summaries, CSM components, final evaluation and runtime information without producing tens of thousands of terminal lines.

## Download citation datasets

```bash
python scripts/download_planetoid.py --dataset cora --output-dir data/raw
python scripts/download_planetoid.py --dataset citeseer --output-dir data/raw
python scripts/download_planetoid.py --dataset pubmed --output-dir data/raw
```

This helper uses PyTorch Geometric for downloading the raw citation networks when that optional package is available. The original AutoGAD/CoLA preprocessing should be treated as the reference when preparing final reproduction datasets.

## Generate injected anomalies

The paper states that the injected datasets follow the ANEMONE/CoLA protocol. The included injector mirrors the public CoLA logic: split randomly selected nodes into structural and attribute anomalies, make small groups fully connected for structural anomalies, and replace attribute vectors by the most distant candidate among a random candidate set for contextual/attribute anomalies.

```bash
python scripts/inject_cola_style.py \
  --input data/processed/cora_clean.mat \
  --output data/processed/cora.mat \
  --m 15 --n 5 --candidate-k 50 --seed 1
```

## Project layout

```text
AutoGAD-Reproduction/
├── configs/                    # paper-aligned experiment/search definitions
├── data/                       # raw + processed data (gitignored)
├── outputs/                    # checkpoints/results/figures (gitignored)
├── scripts/                    # executable workflows
├── src/autogad_reproduction/
│   ├── autogad/                # CSM + generic search
│   ├── models/                 # model interface + ANEMONE + adapters
│   ├── data.py                 # graph loading and validation
│   ├── evaluation.py           # ROC-AUC and paper gain metrics
│   ├── experiments.py          # orchestration
│   └── cli.py                  # command line entry point
├── tests/
├── third_party/                # official repository map, not copied source
├── REPRODUCTION_SPEC.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
└── pyproject.toml
```

## Phase 1 recommended order

1. Run `scripts/smoke_test.py`.
2. Prepare Cora/CiteSeer/PubMed with the CoLA-style injection protocol.
3. Verify native ANEMONE against the authors' official ANEMONE implementation.
4. Run a small ANEMONE AutoGAD search on Cora.
5. Reproduce the sensitivity experiment across `K` and `alpha`.
6. Expand to the full 10-method/10-dataset matrix via the official adapters.

## References

- AutoGAD official repository: https://github.com/ZhongLIFR/AutoGAD2024
- ANEMONE official repository: https://github.com/TrustAGI-Lab/ANEMONE
- ANEMONE paper: https://shiruipan.github.io/publication/cikm-21-jin/cikm-21-jin.pdf
- CoLA official repository: https://github.com/TrustAGI-Lab/CoLA
