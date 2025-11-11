## EvoGrad: Evolved Learning Rules Beyond Backprop

EvoGrad explores learned/evolved learning rules that augment or replace standard backpropagation + Adam. The core idea is to attach small neural modules to layers and the loss that observe local statistics and gradients, generate second-order-like “grad2” signals, and propose parameter updates. These updates can be used alone (Evolved-only) or combined with Adam (Evolved+Adam, hybrid).

The repo includes:
- Evolution loop to search for effective learning-rule parameters
- Baseline vs evolved training comparisons
- Quick and rigorous validation with multiple seeds, datasets, and statistics
- Visualization utilities and example results


### Highlights
- Custom layers (`CustomLinear`, `CustomBias`, `CustomReLU`) with dual gradients (`grad` and `grad2`)
- SPN/GPN modules that transform local data/grad statistics into update signals
- Global NeuroOptimizer that consumes Adam deltas and emits additional evolved updates
- Three training modes: Standard, Evolved-only, Evolved+Adam (hybrid)
- Validation scripts with fair baselines, multiple seeds, and stats (t-tests, effect sizes)


## Installation

1) Python 3.9+ recommended.
2) Install dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib seaborn pandas
```

3) Datasets download automatically via `torchvision` (MNIST, Fashion-MNIST). First run will fetch data into `./data`.


## Quickstart

- Quick validation (tests a strong evolved individual vs baselines on MNIST and Fashion-MNIST):
```bash
python3 quick_validation.py
```

- Lightweight epoch comparison (standard vs evolved hybrid, CSV export):
```bash
python3 simple_epoch_validation.py
```

- Memory-optimized epoch comparison + plots:
```bash
python3 epoch_validation.py
```


## Reproducing Results

Rigorous, publication-style validation across multiple seeds and individuals. Produces JSON/CSV/plots you can analyze or publish.

1) Run rigorous validation:
```bash
python3 rigorous_validation.py
```
Outputs are written to a timestamped directory like:
```
validation_results_YYYYMMDD_HHMMSS/
├── raw_results.json
├── results.csv
├── statistical_analysis.json
└── summary_report.txt
```

2) Build figures and summary tables from results:
```bash
python3 visualize_results.py
```
Saves plots to `visualization_results/`:
- `method_comparison.png` (box plots across methods)
- `statistical_summary.png` (means ± std)
- `training_curves_*.png` (epoch curves)
- `heatmap_*.png` (seed × individual heatmaps)
- `summary_statistics.csv`

For a deeper validation walkthrough see `VALIDATION_README.md`.


## Evolution Loop (optional, longer)

If you want to regenerate or evolve new individuals:

- The evolution driver evaluates a population of learning-rule parameter sets against a control baseline, records fitness, and periodically checkpoints population snapshots.

```bash
python3 evo3.py
```
Artifacts:
- `fitness4.csv`: per-generation best/mean/control metrics (plottable via `plot.py`)
- `population4.pkl`: periodic snapshots of evolved individuals

Plot live fitness curves:
```bash
python3 plot.py
```


## How It Works (brief)

- Custom layers expose both standard gradients and a secondary gradient channel (`grad2`).
- SPN (stat processing net) ingests local statistics (means/std/min/max/skew/kurtosis) of tensors involved in forward/backward for a layer.
- GPN (gradient processing net) mixes local grads, second grads, and SPN features to produce deltas that adjust gradients or propose updates.
- A global NeuroOptimizer consumes Adam deltas and produces additional evolved updates; in hybrid mode we add both.
- Three training modes used in experiments:
  - Standard: backprop + Adam only
  - Evolved-only: evolved updates only (sanity check that rules work standalone)
  - Evolved+Adam: hybrid (original implementation)


## Repository Structure

- `test6.py`
  - Core EvoGrad implementation: custom layers (`CustomLinear`, `CustomBias`, `CustomReLU`), SPN/GPN modules, global `NeuroOptimizer`, computational graph, `NetCustom` and `NetBuilder`, and a minimal `MNISTDataLoader`.
- `evo3.py`
  - Evolutionary search: generates/evaluates populations of learning-rule parameters, performs selection/crossover/mutation, logs to `fitness4.csv`, checkpoints to `population4.pkl`.
- `evaluate_population.py`
  - Baseline vs evolved training across individuals; reports accuracies and ranks.
- `simple_epoch_validation.py`
  - Lightweight standard vs evolved-hybrid comparison, exports CSV with batch/accuracy traces.
- `epoch_validation.py`
  - Memory-optimized epoch comparison (standard vs evolved-hybrid) plus plot creation.
- `quick_validation.py`
  - Tests a chosen evolved individual (default: index 16) across MNIST and Fashion-MNIST under three modes: Standard, Evolved-only, Evolved+Adam. Reports means/std and t-tests.
- `rigorous_validation.py`
  - Publication-grade validation: screens individuals, tests top-N across datasets and multiple seeds, writes JSON/CSV summaries and significance stats.
- `visualize_results.py`
  - Turns rigorous results into publication-ready plots and summary tables in `visualization_results/`.
- `validation_config.py`
  - Centralized knobs for datasets, seeds, batch sizes, epochs, and output toggles.
- `adam.py`
  - Standalone `CustomAdam` reference and comparison against `torch.optim.Adam` on synthetic data.
- `plot.py`, `testing_plot.py`
  - Plot helpers for `fitness4.csv` and `control_data.csv` (moving averages, deltas vs control).
- `VALIDATION_README.md`
  - A focused guide to the validation scripts, design choices, and expected outputs.
- Artifacts
  - `population4.pkl`, `fitness4.csv`, `*_progress.json`, `evolved_model_checkpoint.pth`, figures and results under `visualization_results/` and `validation_results_*/`.


## Typical Workflows

- Compare methods quickly:
  - `python3 quick_validation.py`
- Full, rigorous validation for a paper:
  - `python3 rigorous_validation.py` → `python3 visualize_results.py`
- Evolve new candidates (longer):
  - `python3 evo3.py` and monitor with `python3 plot.py`
- Lightweight smoke test:
  - `python3 simple_epoch_validation.py` (CSV export)


## Tips and Troubleshooting

- Memory
  - Use the memory-optimized scripts (`epoch_validation.py`, `simple_epoch_validation.py`) on limited GPUs/CPUs.
  - Lower batch sizes (`validation_config.py`), reduce number of batches/epochs.
- Reproducibility
  - Scripts use fixed seeds where relevant and reuse baseline weights across methods for fair comparisons.
- Datasets
  - MNIST and Fashion-MNIST download automatically; ensure internet access on first run.
- Artifacts
  - Results and plots are timestamped; keep `validation_results_*` and `visualization_results/` for reporting.


## Research Notes

- Ablations in the validation scripts separate the contribution of evolved updates from Adam.
- Multiple seeds and t-tests reduce the chance improvements are due to noise.
- The Fashion-MNIST runs probe generalization beyond the training domain.

If you need a narrative deep-dive (motivation, hypothesis, method), see your manuscript notes (Beyond Backprop) and adapt key paragraphs into the Overview for your paper/readme audience.


## Citation

If you use EvoGrad in academic work, please cite this repository. A BibTeX entry can be added once a preprint/paper is available.


## License

Specify your license (e.g., MIT) here if you plan to open-source formally.


