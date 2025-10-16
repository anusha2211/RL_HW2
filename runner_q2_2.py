# runner_q2_2.py
from __future__ import annotations
import csv, time
from pathlib import Path
import pandas as pd
from es_trainer import ESConfig, train_es  # adjust 

OUT_CSV = Path("q2_2_runs.csv")

#Fixed "best" hyperparameters for Q2.2
POPULATION  = 45
SIGMA       = 0.15
ALPHA       = 0.15
HIDDEN_SIZES = (4,)          # (tuple) your 1-layer MLP with 4 neurons

#ES runtime knobs
TEMPERATURE       = 0.1
EPISODES_PER_EVAL = 15
ITERATIONS        = 70        # You noticed most good runs converge by ~70
USE_TOPK          = True
ELITES_K          = 5
USE_SHAPING       = False

# 15 independent runs for Q2.2
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 122, 133, 144, 155, 166]

def main():
    first = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        wr = csv.writer(f)
        if first:
            wr.writerow([
                "seed","iter",
                "population","sigma","alpha","hidden_sizes",
                "episodes_per_eval","iterations","temperature",
                "topk","K",
                "J_curr","J_best_true","SR_curr","MX_curr","wallclock_s"
            ])

        print(f"[Q2.2] Running 15 seeds for: P={POPULATION}, sigma={SIGMA}, alpha={ALPHA}, h={HIDDEN_SIZES}")
        for seed in SEEDS:
            cfg = ESConfig(
                neurons_per_layer=HIDDEN_SIZES,
                temperature=TEMPERATURE,
                episodes_per_eval=EPISODES_PER_EVAL,
                iterations=ITERATIONS,
                master_seed=seed,
                population=POPULATION,
                sigma=SIGMA,
                alpha=ALPHA,
                use_topk=USE_TOPK,
                elites=ELITES_K,
                use_shaping=USE_SHAPING,
                parallel=True,
                num_workers=None,
            )
            t0 = time.time()
            out = train_es(cfg)
            hist = out["history"]
            elapsed = time.time() - t0

            for h in hist:
                wr.writerow([
                    seed, h["iter"],
                    POPULATION, SIGMA, ALPHA, HIDDEN_SIZES,
                    EPISODES_PER_EVAL, ITERATIONS, TEMPERATURE,
                    USE_TOPK, ELITES_K,
                    h["J_curr"], h.get("J_best_true",""), h["SR_curr"], h["MX_curr"], f"{elapsed:.1f}"
                ])
            print(f"[Q2.2] seed={seed}  J_final={hist[-1]['J_curr']:.1f}  time={elapsed:.1f}s")

    print("\n[Q2.2] wrote:", OUT_CSV.resolve())

if __name__ == "__main__":
    main()
