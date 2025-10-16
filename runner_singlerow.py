#python runner_final.py --label fast
# runner_final.py
# Re-run selected combos with N seeds; no external CSV required.

from __future__ import annotations
import csv, time, argparse
from pathlib import Path
from typing import List, Dict, Any
from es_trainer import ESConfig, train_es  # uses your shared trainer


# EDIT COMBOS HERE
# hidden_sizes: list[int] (e.g., [4] or [3,2])
COMBOS: List[Dict[str, Any]] = [
    {"label": "good1",   "population": 45, "sigma": 0.175, "alpha": 0.15, "hidden_sizes": [4]},
    {"label": "good2",       "population": 60, "sigma": 0.2, "alpha": 0.1, "hidden_sizes": [4]}, #good2
    {"label": "low_sig_1",   "population": 60, "sigma": 0.1, "alpha": 0.15, "hidden_sizes": [3,2]},
    {"label": "low_sig_2",       "population": 30, "sigma": 0.1, "alpha": 0.1, "hidden_sizes": [3]},
    {"label": "high_sig_2",   "population": 30, "sigma": 0.2, "alpha": 0.1, "hidden_sizes": [3,2]},
    {"label": "low_al_1",       "population": 60, "sigma": 0.15, "alpha": 0.1, "hidden_sizes": [4]},
    {"label": "low_al_2",   "population": 30, "sigma": 0.15, "alpha": 0.1, "hidden_sizes": [3,2]},
    {"label": "high_al_1",       "population": 60, "sigma": 0.1, "alpha": 0.2, "hidden_sizes": [4]},
    {"label": "high_al_2",   "population": 45, "sigma": 0.175, "alpha": 0.2, "hidden_sizes": [3,2]},
    {"label": "moderate_1",       "population": 60, "sigma": 0.15, "alpha": 0.15, "hidden_sizes": [4]},
    {"label": "Moderate_2",   "population": 45, "sigma": 0.15, "alpha": 0.15, "hidden_sizes": [4]},
]


# Default ES settings (override via CLI if needed)
TEMPERATURE       = 0.1
EPISODES_PER_EVAL = 15
ITERATIONS        = 70
USE_TOPK          = True
ELITES_K          = 5
USE_SHAPING       = False
SEEDS_DEFAULT     = [11, 22, 33, 44, 55]

OUT_CSV = Path("final_runs.csv")


# Helpers
def make_combo_id(population: int, sigma: float, alpha: float, hidden_sizes: List[int]) -> str:
    hs = f"({','.join(map(str, hidden_sizes))})" if len(hidden_sizes) > 1 else f"({hidden_sizes[0]})"
    return f"P={population}|sigma={sigma}|alpha={alpha}|h={hs}"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ES for inline-defined combos (no CSV).")
    scope = p.add_mutually_exclusive_group(required=False)
    scope.add_argument("--label", type=str, help='Run only the combo with this label (e.g., "fast")')
    scope.add_argument("--index", type=int, help="Run only the combo at this 0-based index in COMBOS")
    p.add_argument("--iterations", type=int, default=ITERATIONS, help=f"ES iterations (default: {ITERATIONS})")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT, help=f"Seeds list (default: {SEEDS_DEFAULT})")
    p.add_argument("--out", type=str, default=str(OUT_CSV), help=f"Output CSV path (default: {OUT_CSV})")
    p.add_argument("--temp", type=float, default=TEMPERATURE, help=f"Policy temperature (default: {TEMPERATURE})")
    p.add_argument("--episodes", type=int, default=EPISODES_PER_EVAL, help=f"Episodes per eval (default: {EPISODES_PER_EVAL})")
    p.add_argument("--topk", action="store_true", default=USE_TOPK, help="Use top-K ES (default on)")
    p.add_argument("--no-topk", action="store_true", help="Disable top-K; use rank-based ES")
    p.add_argument("--elites", type=int, default=ELITES_K, help=f"K for top-K (default: {ELITES_K})")
    p.add_argument("--shaping", action="store_true", default=USE_SHAPING, help="Enable shaping (default off)")
    return p.parse_args()

def select_combos(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.label:
        matches = [c for c in COMBOS if c.get("label") == args.label]
        if not matches:
            raise ValueError(f'No combo with label="{args.label}". Available: {[c["label"] for c in COMBOS]}')
        return matches
    if args.index is not None:
        if args.index < 0 or args.index >= len(COMBOS):
            raise IndexError(f"--index {args.index} out of range [0, {len(COMBOS)-1}]")
        return [COMBOS[args.index]]
    return COMBOS[:]  # all

def main():
    args = parse_args()
    # Resolve top-k flag combo
    use_topk = args.topk and not args.no_topk

    # Pick which combos to run
    chosen = select_combos(args)
    out_csv_path = Path(args.out)

    # Prepare writer
    first = not out_csv_path.exists()
    with open(out_csv_path, "a", newline="") as f:
        wr = csv.writer(f)
        if first:
            wr.writerow([
                "label","combo_id","seed","iter",
                "population","sigma","alpha","hidden_sizes",
                "episodes_per_eval","iterations","temperature",
                "topk","K",
                "J_curr","J_best_true","SR_curr","MX_curr","wallclock_s"
            ])

        for c in chosen:
            label         = c["label"]
            population    = int(c["population"])
            sigma         = float(c["sigma"])
            alpha         = float(c["alpha"])
            hidden_sizes  = list(map(int, c["hidden_sizes"]))
            combo_id      = make_combo_id(population, sigma, alpha, hidden_sizes)

            print(f"\n[final] {label}: {combo_id}")
            print(f"  iterations={args.iterations}  seeds={args.seeds}")

            for seed in args.seeds:
                cfg = ESConfig(
                    neurons_per_layer=tuple(hidden_sizes),
                    temperature=args.temp,
                    episodes_per_eval=args.episodes,
                    iterations=args.iterations,
                    master_seed=seed,
                    population=population,
                    sigma=sigma,
                    alpha=alpha,
                    use_topk=use_topk,
                    elites=args.elites,
                    use_shaping=args.shaping,
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
                        population, sigma, alpha, tuple(hidden_sizes),
                        args.episodes, args.iterations, args.temp,
                        use_topk, args.elites,
                        h["J_curr"], h.get("J_best_true",""), h["SR_curr"], h["MX_curr"], f"{elapsed:.1f}"
                    ])
                print(f"[final] seed={seed}  J_final={hist[-1]['J_curr']:.1f}  time={elapsed:.1f}s")

    print("\n[final] wrote:", out_csv_path.resolve())

if __name__ == "__main__":
    main()
