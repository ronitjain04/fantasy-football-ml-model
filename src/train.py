import os, json
from pathlib import Path
import pandas as pd
import numpy as np

from etl import load_all
from features import build_feature_table, FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMNS
from model import run_models

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def main():
    df = load_all()
    feats = build_feature_table(df)

    X = feats[[c for c in FEATURE_COLUMNS if c in feats.columns]].fillna(0).to_numpy()
    y = feats[TARGET_COLUMN].to_numpy()

    results = run_models(X, y)
    best = max(results, key=lambda r: r.spearman_rho if r.spearman_rho is not None else -1)

    # Save metrics and outputs
    metrics = {r.name: {"mae": r.mae, "rmse": r.rmse, "spearman_rho": r.spearman_rho} for r in results}
    with open(ARTIFACTS/"metrics.json","w") as f:
        json.dump({"metrics": metrics, "best_model": best.name, "n_samples": len(y), "features": [c for c in FEATURE_COLUMNS if c in feats.columns]}, f, indent=2)

    out_cols = [c for c in ID_COLUMNS + [c for c in FEATURE_COLUMNS if c in feats.columns] + [TARGET_COLUMN] if c in feats.columns]
    out = feats[out_cols].copy()
    out["predicted_points"] = best.preds
    out.to_csv(ARTIFACTS/"predictions.csv", index=False)

    # rankings
    if "season" not in out.columns:
        out["season"] = out.index*0 + 2025
    out["rank"] = out.groupby("season")["predicted_points"].rank(ascending=False, method="dense")
    keep_cols = [c for c in ["season","rank","player","position","team","predicted_points","adp","ppg","ppg_consistency","boom_rate","bust_rate"] if c in out.columns]
    ranks = out[keep_cols].sort_values(["season","rank","player"]).reset_index(drop=True)
    ranks.to_csv(ARTIFACTS/"rankings.csv", index=False)

    print("Wrote artifacts/metrics.json, artifacts/predictions.csv, artifacts/rankings.csv")

if __name__ == "__main__":
    main()
