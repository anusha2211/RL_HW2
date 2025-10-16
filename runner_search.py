# runner_search.py
# Stage A: coarse sweep over hyperparameters (3 seeds per combo).
from __future__ import annotations
import csv, time, itertools, os
from pathlib import Path
from es_trainer import ESConfig, train_es

# ---- Grid to sweep (adjust as needed) ----
POP_LIST   = [30, 45, 60]
SIGMA_LIST = [0.10, 0.15, 0.20]
ALPHA_LIST = [0.10, 0.15]
ARCH_LIST  = [(3,), (4,), (3,2)]

EPISODES_PER_EVAL = 15
ITERATIONS        = 100
TEMPERATURE       = 0.1
USE_TOPK          = True
ELITES_K          = 5
USE_SHAPING       = False

SEEDS = [101, 202, 303]   # 3 seeds

OUT_CSV     = Path("search_runs.csv").resolve()
SUMMARY_CSV = Path("search_summary.csv").resolve()

def cid(P, SIG, ALP, ARCH):
    return f"P={P}|sigma={SIG}|alpha={ALP}|h={tuple(ARCH)}"

def main():
    print("[search] CWD:", os.getcwd())
    print("[search] writing:", OUT_CSV)
    first_runs = not OUT_CSV.exists()
    first_summ = not SUMMARY_CSV.exists()

    with open(OUT_CSV, "a", newline="") as f_runs, open(SUMMARY_CSV, "a", newline="") as f_summ:
        wr = csv.writer(f_runs)
        ws = csv.writer(f_summ)

        if first_runs:
            wr.writerow([
                "combo_id","seed","iter",
                "population","sigma","alpha","hidden_sizes",
                "episodes_per_eval","iterations","temperature",
                "topk","K",
                "J_curr","J_best_true","SR_curr","MX_curr","wallclock_s"
            ])
        if first_summ:
            ws.writerow([
                "combo_id","seed","population","sigma","alpha","hidden_sizes",
                "episodes_per_eval","iterations",
                "J_final","first_iter_ge_-150","wallclock_s_total"
            ])

        for P, SIG, ALP, ARCH in itertools.product(POP_LIST, SIGMA_LIST, ALPHA_LIST, ARCH_LIST):
            combo = cid(P, SIG, ALP, ARCH)
            print(f"\n[search] combo: {combo}")
            for seed in SEEDS:
                cfg = ESConfig(
                    neurons_per_layer=tuple(ARCH),
                    temperature=TEMPERATURE,
                    episodes_per_eval=EPISODES_PER_EVAL,
                    iterations=ITERATIONS,
                    master_seed=seed,
                    population=P,
                    sigma=SIG,
                    alpha=ALP,
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
                        combo, seed, h["iter"],
                        P, SIG, ALP, tuple(ARCH),
                        EPISODES_PER_EVAL, ITERATIONS, TEMPERATURE,
                        USE_TOPK, ELITES_K,
                        h["J_curr"], h.get("J_best_true",""), h["SR_curr"], h["MX_curr"], ""
                    ])

                J_final = hist[-1]["J_curr"]
                first_ge = next((h["iter"] for h in hist if h["J_curr"] >= -150), None)
                ws.writerow([combo, seed, P, SIG, ALP, tuple(ARCH),
                             EPISODES_PER_EVAL, ITERATIONS, J_final, first_ge, f"{elapsed:.1f}"])
                print(f"[search] seed={seed}  J_final={J_final:.1f}  first>=-150 @iter={first_ge}  time={elapsed:.1f}s")

    print("[search] done.")

if __name__ == "__main__":
    main()
