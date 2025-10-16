# plot_q2_2.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = Path("q2_2_runs.csv")
OUT_PNG = Path("q2_2_best_combo.png")
MAX_ITER_PLOT = 70  # keep consistent with runner

def main():
    df = pd.read_csv(IN_CSV)

    # Keep only the first MAX_ITER_PLOT iterations 
    df = df[df["iter"] <= MAX_ITER_PLOT].copy()

    # Per-iteration mean and std over the 15 seeds
    agg = df.groupby("iter")["J_curr"].agg(["mean", "std"]).reset_index()

    # Grab the (unique) HPs for the title
    hp = df.iloc[0]
    title = f"P={int(hp['population'])}, σ={hp['sigma']}, α={hp['alpha']}, h={hp['hidden_sizes']} (15 runs)"

    # Plot
    plt.figure(figsize=(7.2, 4.6))

    # (Optional) overlay individual seed curves, faint
    for seed, g in df.groupby("seed"):
        g = g.sort_values("iter")
        plt.plot(g["iter"], g["J_curr"], linewidth=0.8, alpha=0.25)

    # Mean curve and std band
    plt.plot(agg["iter"], agg["mean"], linewidth=2.0, label="Mean (15 runs)")
    plt.fill_between(agg["iter"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.25, label="±1 std")

    plt.axhline(-150, linestyle="--", linewidth=1.0)  # visual reference band
    plt.axhline(-120, linestyle="--", linewidth=1.0)

    plt.title(title)
    plt.xlabel("ES iterations")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("[Q2.2] saved figure:", OUT_PNG.resolve())

if __name__ == "__main__":
    main()
