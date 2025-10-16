#Identify 6 combos
'''Reads search_summary.csv. 
Picks 6 combos (baseline, fast, stable, two_layer, one_layer, surprising) like before. 
Writes them to selected_combos.csv so the next script can re-run them with 5 seeds.'''

# analyze_search_identify.py
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

SUMMARY_CSV = "search_summary.csv"
OUT_SELECTED = Path("selected_combos.csv")

def parse_hidden_sizes(hs):
    if isinstance(hs, tuple): return hs
    txt = str(hs).strip()
    try:
        val = eval(txt, {}, {})
        if isinstance(val, tuple): return val
    except Exception:
        pass
    return tuple(int(x) for x in txt.replace("(", "").replace(")", "").split(",") if x.strip())

def normalize(df):
    df = df.copy()
    df["hidden_sizes_tuple"] = df["hidden_sizes"].apply(parse_hidden_sizes)
    df["ok"] = df["J_final"] >= -150.0
    return df

def summarize(df):
    g = (df.groupby(["population","sigma","alpha","hidden_sizes_tuple"], as_index=False)
           .agg(n=("J_final","size"),
                meanJ=("J_final","mean"),
                stdJ=("J_final","std"),
                ok_rate=("ok","mean"),
                first_hit_median=("first_iter_ge_-150", lambda s: np.nanmedian(s.values))))
    g["combo_id"] = ("P=" + g["population"].astype(str) +
                     "|sigma=" + g["sigma"].astype(str) +
                     "|alpha=" + g["alpha"].astype(str) +
                     "|h=" + g["hidden_sizes_tuple"].astype(str))
    return g

def pick_baseline(agg):
    cand = agg[(agg["ok_rate"] >= (2/3)) & (agg["meanJ"].between(-160, -135))]
    if len(cand)==0:
        cand = agg[(agg["ok_rate"] >= 0.5)].copy()
        cand = cand.iloc[(cand["meanJ"] - (-150)).abs().argsort()]
    return cand.head(1)

def pick_fast(agg, used):
    cand = agg[(agg["ok_rate"] >= (2/3)) & (~agg["combo_id"].isin(used))].copy()
    cand = cand.dropna(subset=["first_hit_median"])
    if len(cand)==0: return cand.head(0)
    return cand.sort_values(["first_hit_median","meanJ"], ascending=[True,False]).head(1)

def pick_stable(agg, used):
    cand = agg[(agg["ok_rate"] >= 1.0) & (~agg["combo_id"].isin(used))].copy()
    cand = cand.dropna(subset=["stdJ"])
    if len(cand)==0: return cand.head(0)
    return cand.sort_values(["stdJ","meanJ"], ascending=[True,False]).head(1)

def pick_two_layer(agg, used):
    cand = agg[(agg["hidden_sizes_tuple"].astype(str)=="(3, 2)") &
               (agg["ok_rate"] >= (2/3)) &
               (~agg["combo_id"].isin(used))].copy()
    return cand.sort_values("meanJ", ascending=False).head(1)

def pick_one_layer_match(agg, two_row, used):
    if len(two_row)==0: return two_row.head(0)
    p,s,a = two_row.iloc[0][["population","sigma","alpha"]]
    cand = agg[(agg["population"]==p) &
               (agg["sigma"]==s) &
               (agg["alpha"]==a) &
               (agg["hidden_sizes_tuple"].astype(str).isin(["(4,)","(3,)"])) &
               (~agg["combo_id"].isin(used))].copy()
    if len(cand)==0:
        cand = agg[(agg["hidden_sizes_tuple"].astype(str).isin(["(4,)","(3,)"])) &
                   (agg["ok_rate"] >= (2/3)) &
                   (~agg["combo_id"].isin(used))].copy()
        if len(cand)==0: return cand.head(0)
    return cand.sort_values("meanJ", ascending=False).head(1)

def pick_surprising(agg, used):
    cand = agg[~agg["combo_id"].isin(used)].copy()
    cand["pitfall"] = (1 - cand["ok_rate"])*2 + np.nan_to_num(cand["first_hit_median"], nan=999)/100.0
    return cand.sort_values("pitfall", ascending=False).head(1)

def main():
    df = pd.read_csv(SUMMARY_CSV)
    df = normalize(df)
    agg = summarize(df)

    used = set()
    picks = []

    lbl,row = "baseline", pick_baseline(agg);    picks.append((lbl,row))
    used.update(row["combo_id"].tolist())

    lbl,row = "fast", pick_fast(agg,used);       picks.append((lbl,row))
    used.update(row["combo_id"].tolist())

    lbl,row = "stable", pick_stable(agg,used);   picks.append((lbl,row))
    used.update(row["combo_id"].tolist())

    lbl,row = "two_layer", pick_two_layer(agg,used); picks.append((lbl,row))
    used.update(row["combo_id"].tolist())

    lbl,row = "one_layer", pick_one_layer_match(agg,row,used); picks.append((lbl,row))
    used.update(row["combo_id"].tolist())

    lbl,row = "surprising", pick_surprising(agg,used); picks.append((lbl,row))

    out_rows = []
    for label, r in picks:
        if len(r)==0: continue
        r = r.iloc[0]
        out_rows.append(dict(
            label=label,
            combo_id=r["combo_id"],
            population=int(r["population"]),
            sigma=float(r["sigma"]),
            alpha=float(r["alpha"]),
            hidden_sizes=str(tuple(r["hidden_sizes_tuple"])),
            meanJ=float(r["meanJ"]),
            stdJ=float(r["stdJ"]) if not np.isnan(r["stdJ"]) else "",
            ok_rate=float(r["ok_rate"]),
            first_hit_median=("" if np.isnan(r["first_hit_median"]) else int(r["first_hit_median"]))
        ))

    pd.DataFrame(out_rows).to_csv(OUT_SELECTED, index=False)
    print("Wrote:", OUT_SELECTED.resolve())
    print(pd.DataFrame(out_rows))

if __name__ == "__main__":
    main()
