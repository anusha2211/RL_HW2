# runner_final.py
# rerun 6 combos with 5 seeds
'''
Reads selected_combos.csv. 
Runs each of the 6 combos with 5 seeds 
Writes per-iteration rows to final_runs.csv (so we can plot 5 lines + mean).
'''
# runner_final.py
from __future__ import annotations
import csv, time
from pathlib import Path
import pandas as pd
from es_trainer import ESConfig, train_es  

SELECTED = Path("selected_combos.csv")
OUT_CSV  = Path("final_runs.csv")

TEMPERATURE       = 0.1
EPISODES_PER_EVAL = 15
ITERATIONS        = 70        # Q2.1 can be 100; 
USE_TOPK          = True       # to match PDF’s top-K wording
ELITES_K          = 5
USE_SHAPING       = False
SEEDS             = [11, 22, 33, 44, 55]   # 5 runs per combo

def main():
    combos = pd.read_csv(SELECTED)
    first = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        wr = csv.writer(f)
        if first:
            wr.writerow([
                "label","combo_id","seed","iter",
                "population","sigma","alpha","hidden_sizes",
                "episodes_per_eval","iterations","temperature",
                "topk","K",
                "J_curr","J_best_true","SR_curr","MX_curr","wallclock_s"
            ])

        for _, row in combos.iterrows():
            label = row["label"]
            combo_id = row["combo_id"]
            population = int(row["population"])
            sigma = float(row["sigma"])
            alpha = float(row["alpha"])
            hidden_sizes = eval(row["hidden_sizes"], {}, {})  # "(3,)" -> (3,)

            print(f"\n[final] {label}: {combo_id}")
            for seed in SEEDS:
                cfg = ESConfig(
                    neurons_per_layer=hidden_sizes,
                    temperature=TEMPERATURE,
                    episodes_per_eval=EPISODES_PER_EVAL,
                    iterations=ITERATIONS,
                    master_seed=seed,
                    population=population,
                    sigma=sigma,
                    alpha=alpha,
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
                        label, combo_id, seed, h["iter"],
                        population, sigma, alpha, hidden_sizes,
                        EPISODES_PER_EVAL, ITERATIONS, TEMPERATURE,
                        USE_TOPK, ELITES_K,
                        h["J_curr"], h.get("J_best_true",""), h["SR_curr"], h["MX_curr"], f"{elapsed:.1f}"
                    ])
                print(f"[final] seed={seed}  J_final={hist[-1]['J_curr']:.1f}  time={elapsed:.1f}s")

    print("\n[final] wrote:", OUT_CSV.resolve())

if __name__ == "__main__":
    main()
