#Plot 5 lines + mean
'''
Reads final_runs.csv. 
For each of the 6 labels, draws: 5 individual seed lines (faint), the mean line (bold), optional ±1 std band, an optional x-limit (e.g., 70 iterations).
'''

# plot_final_curves.py
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FINAL_CSV = "final_runs.csv"
OUT_DIR   = Path("final_plots")
MAX_ITERS = 70     # set to None to show all iterations
SHOW_STD  = True   # shade ±1 std around the mean

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(FINAL_CSV)

    # Basic sanity
    needed = {"label","combo_id","seed","iter","J_curr"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"final_runs.csv is missing columns: {missing}")

    for label, sub in df.groupby("label"):
        if MAX_ITERS is not None:
            sub = sub[sub["iter"] <= MAX_ITERS]

        # individual seeds
        plt.figure(figsize=(7,4))
        for seed, g in sub.groupby("seed"):
            g = g.sort_values("iter")
            plt.plot(g["iter"], g["J_curr"], alpha=0.35, linewidth=1.0, label=f"seed {seed}")

        # mean ± std
        grp = (sub.groupby("iter", as_index=False)
                 .agg(meanJ=("J_curr","mean"), stdJ=("J_curr","std")))
        xs = grp["iter"].to_numpy()
        mu = grp["meanJ"].to_numpy()
        sd = grp["stdJ"].fillna(0.0).to_numpy()
        if SHOW_STD:
            plt.fill_between(xs, mu - sd, mu + sd, alpha=0.2, label="±1 std")
        plt.plot(xs, mu, linewidth=2.5, label="mean across 5 runs")

        plt.axhline(-150, ls="--", label="acceptable (~ -150)")
        plt.axhline(-120, ls="--", label="near-optimal (~ -120)")
        if MAX_ITERS is not None:
            plt.xlim(0, MAX_ITERS)

        # Put combo hyperparams in the title (they’re constant per label)
        one = sub.iloc[0]
        title = (f"{label}: P={one['population']}, σ={one['sigma']}, α={one['alpha']}, h={one['hidden_sizes']}")
        plt.title(title)
        plt.xlabel("ES iteration")
        plt.ylabel("Return (↑ better)")
        plt.legend(loc="lower right", fontsize=8, ncol=2)
        plt.tight_layout()

        out_png = OUT_DIR / f"{label}.png"
        plt.savefig(out_png)
        plt.close()
        print("saved:", out_png.resolve())

if __name__ == "__main__":
    main()
