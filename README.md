# RL_HW2 — Evolution Strategies (ES) Experiments

This repository contains my implementation and analysis for **HW2 (Programming)**, using **Evolution Strategies (ES)** to optimize a policy and study how hyperparameters affect learning curves.

It includes:
- ES training code (`es_trainer.py`)
- Runners to reproduce the **5-seed** category plots and the **15-run** best-config plot
- CSV logs for analysis and the final figures used in the write-up

> **TL;DR**: run `runner_final.py` to reproduce results; logs are written to CSV and plots can be generated from those logs.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Requirements](#requirements)
- [Project Layout](#project-layout)
- [What Each File Does](#what-each-file-does)
- [How to Run](#how-to-run)
  - [A) Reproduce 5-seed learning curves for selected combos](#a-reproduce-5-seed-learning-curves-for-selected-combos)
  - [B) Reproduce the 15-run “best config” curve (Q2.2)](#b-reproduce-the-15-run-best-config-curve-q22)
- [Plotting Tips](#plotting-tips)
- [Repro Notes](#repro-notes)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citation / Attribution](#citation--attribution)

---

## Environment Setup

Use a clean virtual environment (conda or venv).

```bash
# Using conda
conda create -n rl-hw2 python=3.10 -y
conda activate rl-hw2

# Or with venv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```

---

## Requirements

Save the following as **`requirements.txt`** in the repo root:

```txt
# Core
numpy>=1.24
pandas>=2.1
matplotlib>=3.7
tqdm>=4.66

# RL env (pick one your code uses)
gymnasium>=0.29
# If you use classic Gym instead, comment the line above and use:
# gym==0.26.2

# Optional extras for specific environments (uncomment if needed)
# gymnasium[box2d]>=0.29
# mujoco>=3.0.0
# joblib>=1.3
```

Then install:
```bash
pip install -r requirements.txt
```

If your environment requires extras (e.g., Box2D / Mujoco), add them to `requirements.txt` and reinstall.

---

## Project Layout

```
.
├─ es_trainer.py               # ESConfig dataclass + train_es(cfg) — core ES training loop
├─ env.py                      # (optional) Env factory/wrappers, seeding, normalization
├─ policy.py                   # Policy network, param vectorization helpers
├─ main.py                     # Single-run entry point for quick sanity checks
│
├─ Runner_search.py            # Coarse grid search over ~54 combos → search_summary.csv
├─ Analyze_search_identify.py  # (optional) Auto-pick representative configs → selected_combos.csv
├─ Runner_final.py             # Re-run selected combos with 5 seeds → final_runs.csv
├─ Runner_singlerow.py         # Run a single combo for 5 seeds → final_runs.csv
├─ Runner_q2_2.py              # Run best config over 15 seeds → final_runs.csv
│
├─ Plot_final_curves.py        # Build Q2.1 figures from final_runs.csv
├─ Plot_q2_2.py                # Build Q2.2 figure from aggregated runs
│
├─ search_summary.csv          # (generated) Grid-search summary across combos
├─ selected_combos.csv         # (generated/edited) Curated combos for Q2.1
├─ final_runs.csv              # (generated) Per-iteration logs for each run (appended)
│
├─ final_plots/                    # Folder to collect figures 
│  └─ *.png                    # Plots used in the report 
├─ *.png                       # Plots
└─ README.md
```

---

## What Each File Does

Below is a quick, practical map of the codebase: what each script/module is for, what it reads/writes, and how it plugs into the workflow.

---

### `Analyze_search_identify.py`  _(optional / extra)_
**Purpose:** Post-process the coarse grid search to auto-pick 6 illustrative configs (e.g., `baseline`, `fast`, `stable`, `two_layer`, `one_layer`, `surprising`) for re-running with 5 seeds.

- **Reads:** `search_summary.csv` (one row per run from your grid search)
- **Writes:** `selected_combos.csv` with columns like  
  `label,combo_id,population,sigma,alpha,hidden_sizes`
- **When to use:** After you’ve completed the 54-combo grid search and want a curated set for Q2.1 plots.
- **Example:**
  ```bash
  python Analyze_search_identify.py
  # -> produces selected_combos.csv for the next stage
  ```

---

### `Runner_final.py` _(optional / extra)_
**Purpose:** Re-run each configuration in `selected_combos.csv` with 5 seeds, log per-iteration metrics for plotting “5 lines + mean ± std”.

- **Reads:** `selected_combos.csv`
- **Writes:** Appends to `final_runs.csv` (long-form: one row per (combo × seed × iter))
- **Depends on:** `es_trainer.py` (imports `ESConfig`, `train_es`)
- **When to use:** To produce the learning-curve data for the 5-seed figures in Q2.1.
- **Example:**
  ```bash
  python Runner_final.py
  # optional: python Runner_final.py --row 3   # run only one selected row (if supported)
  ```

---

### `env.py` _(if present)_
**Purpose:** Environment construction/utilities (wrappers, seeding, normalization).  
If your project uses Gym/Gymnasium directly inside `train_es`, `env.py` might be unnecessary.  

If both `env.py` and `policy.py` exist, they serve different roles:
- `env.py` → “world” setup (Gym env factory/wrappers)
- `policy.py` → “brain” (the neural network policy)

**Exports (typical):** `make_env(seed)`, observation/action space helpers.  
**Used by:** `es_trainer.py` and/or `main.py` to create/reset the environment.  

**Note:** If unused, it can be safely removed; otherwise, keep it as the centralized environment factory.

---

### `policy.py`
**Purpose:** Defines the policy network and action-selection logic used by ES.  
**Exports (typical):** `Policy(hidden_sizes, obs_dim, act_dim)`, `forward/infer`, parameter helpers (`get_params()`, `set_params()`).  
**Used by:** `es_trainer.py` inside `train_es` to evaluate perturbed policies.

---

### `main.py`
**Purpose:** Convenience entry-point for single-run experiments (e.g., run ES once with a chosen config to sanity-check changes).  
**Reads:** (none required — hyperparameters hard-coded or passed via CLI)  
**Writes:** Prints/plots a single run or writes a one-off CSV.  
**Depends on:** `es_trainer.py` (imports `ESConfig`, `train_es`)  
**When to use:** Quick debugging before running batch scripts.  
**Example:**
```bash
python main.py --population 45 --sigma 0.15 --alpha 0.15 --hidden "(4,)" --iters 70
```

---

### `es_trainer.py`
**Purpose:** Shared ES implementation so all runners import the same logic.

**Exports:**
- `ESConfig` dataclass (population, sigma, alpha, hidden_sizes, iterations, episodes_per_eval, master_seed, use_topk, etc.)
- `train_es(cfg)` → returns `{"history": [...], "best": ..., "meta": ...}` with fields like `iter`, `J_curr`, `SR_curr`, `MX_curr`, `J_best_true`.

**Used by:** `runner_search.py`, `Runner_final.py`, `Runner_q2_2.py`, `main.py`, plotting pipeline.

**Note:** Keeping ES here avoids code duplication and ensures consistency across runners.

---

### `Runner_search.py`
**Purpose:** Run the coarse grid search over 54 hyperparameter combos.

- **Reads:** (none; combos defined internally or via CLI)
- **Writes:**  
  - Per-run summaries → `search_summary.csv` (e.g., `J_final`, `first_iter_ge_-150`, `wallclock_s_total`)  
  - Optional per-iteration logs (if enabled)
- **When to use:** First stage — to identify promising configurations.
- **Example:**
  ```bash
  python Runner_search.py
  # -> produces search_summary.csv
  ```

---

### `Runner_singlerow.py`
**Purpose:** Run a single hyperparameter combo for 5 seeds (useful for quick category checks without editing CSVs).  
**Inputs:** Hard-coded combo or CLI flags.  
**Writes:** Appends per-iteration lines to `final_runs.csv`.  
**When to use:** To regenerate one figure’s data without re-running all combos.  
**Example:**
```bash
python Runner_singlerow.py --population 45 --sigma 0.15 --alpha 0.15 --hidden "(4,)" --seeds 5
```

---

### `Plot_final_curves.py`
**Purpose:** Create Q2.1 figures from `final_runs.csv`: per-category plots with mean across 5 runs and ±1 std shading.

- **Reads:** `final_runs.csv`
- **Writes:** PNGs like  
  `good1.png, good2.png, low_sig_1.png, low_sig_2.png, low_al_1.png, low_al_2.png, high_al_1.png, high_al_2.png, moderate_1.png, Moderate_2.png`
- **Example:**
  ```bash
  python Plot_final_curves.py
  ```

---

### `Plot_q2_2.py`
**Purpose:** Create the Q2.2 figure — the best hyperparameters evaluated over 15 runs, showing mean ±1 std per iteration.

- **Reads:** `final_runs.csv` or separate CSV from `Runner_q2_2.py`
- **Writes:** `q2_2_best_combo.png`
- **Example:**
  ```bash
  python Plot_q2_2.py
  ```

---

### `Runner_q2_2.py`
**Purpose:** Evaluate the best hyperparameter setting over 15 seeds for Q2.2.  
**Inputs:** Best config (hard-coded or via CLI).  
**Writes:** Appends per-iteration rows for all 15 runs into `final_runs.csv`.  
**When to use:** Before running `Plot_q2_2.py`.  
**Example:**
```bash
python Runner_q2_2.py --population 45 --sigma 0.15 --alpha 0.15 --hidden "(4,)" --seeds 15
```


## How to Run

### A) Reproduce 5-seed learning curves for selected combos (Q2.1)

#### Option A — CSV-driven
Create or edit `selected_combos.csv`:
```csv
label,combo_id,population,sigma,alpha,hidden_sizes
high_sigma_1,P=45|sigma=0.175|alpha=0.15|h=(4,),45,0.175,0.15,"(4,)"
high_sigma_2,P=60|sigma=0.20|alpha=0.10|h=(4,),60,0.20,0.10,"(4,)"
low_sigma_1,P=60|sigma=0.10|alpha=0.15|h=(3,2),60,0.10,0.15,"(3,2)"
low_sigma_2,P=30|sigma=0.10|alpha=0.10|h=(3,),30,0.10,0.10,"(3,)"
```

Run:
```bash
python runner_final.py
```

#### Option B — In-file PARAMS
Edit the `COMBOS` list in `runner_final.py` and run:
```bash
python runner_final.py
```

#### (Optional) Run a single combo
```bash
python runner_final.py --row 2
```

---

### B) Reproduce the 15-run “best config” curve (Q2.2)

**Best config used:**
```
P=45, σ=0.15, α=0.15, h=(4,)
```

#### Approach 1 — Single combo, 15 seeds
- Keep one config  
- Set `SEEDS = [1, 2, ..., 15]`
- Run `python runner_final.py`
- Aggregate per-iteration metrics and plot mean ±1 std

#### Approach 2 — Dedicated block
Add a loop for 15 seeds directly in `runner_final.py`.

---

## Plotting Tips

Group by `(label, iter)` or `(combo_id, iter)` and compute:
- Mean and std of `J_curr` across seeds  
Plot mean curve with ±1 std shading.

Reference lines:
- Acceptable ≈ −150  
- Near-optimal ≈ −120  

---

## Repro Notes

- **Seeds:** [11, 22, 33, 44, 55] for Q2.1; 15 distinct for Q2.2  
- **Append behavior:** `final_runs.csv` is appended to each run  
- **Fresh start:** delete `final_runs.csv` before re-running  
- **Parallel runs:** check worker configs in `ESConfig`  
- **Edit CSVs:** use plain-text editor to avoid quote corruption

---

## Troubleshooting

**CSV corrupted?**  
Keep hidden_sizes quoted like `"(4,)"`.

**Only re-run one config?**  
Use `--row` or comment out others.

**Plots don’t match report?**  
Check averaging method and std shading.

**Figures misplaced in LaTeX?**  
Try `[p]`, `[!htbp]`, or `\FloatBarrier`.

---

## FAQ

**Q:** Where do the PNGs come from?  
**A:** Aggregated from `final_runs.csv` per iteration, averaged across seeds.

**Q:** Can figures live in `/figures`?  
**A:** Yes — adjust LaTeX or plotting paths.

**Q:** Which environment is used?  
**A:** `gymnasium` (default); switch to `gym` if needed.

**Q:** How to change hyperparameters?  
**A:** Edit `COMBOS` list or update `selected_combos.csv`.

**Q:** How to reset logs?  
**A:** Delete `final_runs.csv`.

---

## Citation / Attribution

This repo is for an academic HW on **Evolution Strategies**.  
If you build on this, please cite your course and link to the original assignment.

Libraries used: **NumPy**, **Pandas**, **Matplotlib**, **Gymnasium**, **tqdm**.
