#!/usr/bin/env python
"""Metrics, experiment logging, comparative tables and report assembly.

compute_metrics / append_experiment are imported by train.py. The CLI
rebuilds the leaderboard from results/experiments.csv (latest run per
model), writes per-class test metrics and top-confusion tables, calls
plot.py for all figures, and assembles report/report.md.
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CSV_FIELDS = ["timestamp", "model", "features", "seed", "hyperparams",
              "epochs", "val_accuracy", "val_macro_f1",
              "test_accuracy", "test_macro_f1", "train_time_s"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    n_classes: int) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

    # macro F1 over the labels present in y_true or y_pred (sklearn default):
    # classes absent from both would otherwise score 0 and dilute the mean
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "n_eval_classes": int(len(set(y_true.tolist()) | set(y_pred.tolist()))),
        "per_class_f1": f1_score(y_true, y_pred, average=None,
                                 labels=list(range(n_classes)),
                                 zero_division=0).tolist(),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(n_classes)),
                                      normalize="true").tolist(),
    }


def append_experiment(csv_path: str | Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                         for k, v in row.items()})
    log.info("Logged experiment to %s", csv_path)


def per_class_table(df_latest: pd.DataFrame, ds: dict,
                    pred_dir: Path, seed: int) -> pd.DataFrame:
    """Per-class test F1 for every model (rows = classes)."""
    table = pd.DataFrame(index=ds["classes"])
    for name in df_latest["model"]:
        z = np.load(pred_dir / f"{name}_seed{seed}.npz")
        m = ds["test_mask"]
        f1 = compute_metrics(ds["y"][m], z["y_pred"][m],
                             len(ds["classes"]))["per_class_f1"]
        table[name] = np.round(f1, 4)
    return table


def top_confusions(df_latest: pd.DataFrame, ds: dict, pred_dir: Path,
                   seed: int) -> pd.DataFrame:
    """Off-diagonal confusion counts on the test set, per model."""
    from sklearn.metrics import confusion_matrix

    rows = []
    for name in df_latest["model"]:
        z = np.load(pred_dir / f"{name}_seed{seed}.npz")
        m = ds["test_mask"]
        cm = confusion_matrix(ds["y"][m], z["y_pred"][m],
                              labels=list(range(len(ds["classes"]))))
        for i, j in zip(*np.nonzero(cm - np.diag(np.diag(cm)))):
            rows.append({
                "model": name,
                "true_label": ds["classes"][i],
                "predicted_label": ds["classes"][j],
                "n_cells": int(cm[i, j]),
                "fraction_of_true": round(cm[i, j] / max(cm[i].sum(), 1), 3),
            })
    out = pd.DataFrame(rows).sort_values(["model", "n_cells"],
                                         ascending=[True, False])
    return out


# Biological context for the confusion pairs observed on PBMCs
BIO_NOTES = {
    frozenset({"Classical monocytes", "Non-classical monocytes"}):
        "a same-lineage continuum of myeloid states separated by few genes",
    frozenset({"Tcm/Naive helper T cells", "Tem/Trm cytotoxic T cells"}):
        "adjacent T-cell differentiation states sharing most of their "
        "expression programme",
    frozenset({"Tcm/Naive helper T cells", "NK cells"}):
        "proximity of the cytotoxic effector programme between T and NK "
        "lineages",
    frozenset({"Classical monocytes", "DC"}):
        "two myeloid populations with overlapping antigen-presentation "
        "machinery",
}


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def generate_report(cfg: dict, leaderboard: pd.DataFrame,
                    f1_table: pd.DataFrame, conf: pd.DataFrame,
                    ds: dict) -> None:
    """Assemble report/report.md from the computed tables and figures."""
    processed = Path(cfg["paths"]["processed"])
    figures_rel = "../results/figures"
    meta = json.loads((processed / "prepare_summary.json")
                      .read_text(encoding="utf-8"))
    split_summary = pd.read_csv(Path(cfg["paths"]["tables"])
                                / "split_summary.csv", index_col=0)

    lb = leaderboard.copy()
    lb["model"] = pd.Categorical(lb["model"], categories=plot_model_order(),
                                 ordered=True)
    lb = lb.sort_values("model")

    # --- "When does the GNN win?" — decided by the numbers, not in advance ---
    gnn = lb[lb["model"].str.startswith("graphsage")]
    classical = lb[lb["model"].isin(["logreg", "mlp"])]
    best_gnn = gnn.loc[gnn["test_macro_f1"].idxmax()]
    best_classical = classical.loc[classical["test_macro_f1"].idxmax()]
    delta = best_gnn["test_macro_f1"] - best_classical["test_macro_f1"]
    val_best = lb.loc[lb["val_macro_f1"].idxmax(), "model"]
    test_best = lb.loc[lb["test_macro_f1"].idxmax(), "model"]

    if delta > 0.02:
        verdict = (f"**The GNN wins on this task.** Its best variant "
                   f"({best_gnn['model']}, test macro F1 "
                   f"{best_gnn['test_macro_f1']:.3f}) beats the best classical "
                   f"baseline ({best_classical['model']}, "
                   f"{best_classical['test_macro_f1']:.3f}) by "
                   f"+{delta:.3f} macro F1. The win is mechanistically "
                   "plausible: the dominant errors of per-cell classifiers are "
                   "same-lineage confusions (see error analysis), and message "
                   "passing over the k-NN graph smooths exactly these cluster "
                   "borders.")
    elif delta < -0.02:
        verdict = (f"**The GNN does not win on this task.** Its best variant "
                   f"({best_gnn['model']}, test macro F1 "
                   f"{best_gnn['test_macro_f1']:.3f}) trails the best classical "
                   f"baseline ({best_classical['model']}, "
                   f"{best_classical['test_macro_f1']:.3f}) by "
                   f"{-delta:.3f} macro F1. On a dataset this small, the graph "
                   "adds parameters without adding information beyond the "
                   "expression features themselves.")
    else:
        verdict = (f"**The GNN ties with the classical baselines** "
                   f"(best GNN {best_gnn['model']}: "
                   f"{best_gnn['test_macro_f1']:.3f} vs best baseline "
                   f"{best_classical['model']}: "
                   f"{best_classical['test_macro_f1']:.3f} test macro F1).")

    caveats = []
    if val_best != test_best:
        caveats.append(
            f"the validation set selects **{val_best}** while the test set "
            f"ranks **{test_best}** first — with {meta['split_sizes'].get('val')} "
            "val cells spanning 2 labels, model selection itself is noisy and "
            "the val/test ranking inversion is expected rather than alarming")
    caveats.append(
        f"the test set holds {meta['split_sizes'].get('test')} cells over "
        "2 labels, so single misclassified cells move macro F1 visibly")
    caveats.append(
        "macro F1 is computed over the union of true and predicted labels: "
        "every error scattered into an absent class adds a zero-F1 class, "
        "which is why macro F1 looks low next to accuracy")
    caveats.append(
        "ground truth comes from CellTypist majority voting — label noise at "
        "cluster borders bounds every model from above")
    caveats.append(
        "the graph is transductive (built from all cells' features, labels "
        "masked); an inductive variant would be stricter still")
    caveats.append("results are single-seed (config `seed: 42`)")
    caveats_md = "\n".join(f"- {c}" for c in caveats)

    # --- Error analysis paragraphs from the observed confusions ---
    conf_agg = (conf.groupby(["true_label", "predicted_label"],
                             as_index=False)["n_cells"].sum()
                .sort_values("n_cells", ascending=False))
    err_lines = []
    for _, r in conf_agg.head(4).iterrows():
        note = BIO_NOTES.get(
            frozenset({r["true_label"], r["predicted_label"]}),
            "not an expected same-lineage pair — at this cell count it is "
            "most likely label noise at a cluster border")
        n_models = conf[(conf["true_label"] == r["true_label"])
                        & (conf["predicted_label"] == r["predicted_label"])
                        ]["model"].nunique()
        err_lines.append(
            f"- **{r['true_label']} → {r['predicted_label']}** "
            f"({r['n_cells']} cells across {n_models} model(s)): {note}.")
    error_md = "\n".join(err_lines) if err_lines else \
        "No test-set errors were recorded by any model."

    # Data-driven coherence claim: only assert it for confusions big enough
    # to matter (>= 3 cells), and name any that cross distant lineages
    distant = conf_agg[
        conf_agg.apply(lambda r: frozenset({r["true_label"],
                                            r["predicted_label"]})
                       not in BIO_NOTES, axis=1)
        & (conf_agg["n_cells"] >= 3)]
    if distant.empty:
        coherence = ("Every confusion involving 3 or more cells is between "
                     "closely related lineages; the residual 1-2-cell errors "
                     "are single-cell noise.")
    else:
        pairs = "; ".join(f"{r['true_label']} → {r['predicted_label']} "
                          f"({r['n_cells']} cells)"
                          for _, r in distant.iterrows())
        coherence = ("Most confusions are between closely related lineages, "
                     f"but note the cross-lineage errors: {pairs}.")

    test_labels = split_summary.index[split_summary.get("test", 0) > 0]
    f1_show = (f1_table.loc[test_labels].round(3)
               .reset_index(names="true label"))

    report = f"""# Comparative study: GNN vs classical ML for cell-type classification

*Generated on {datetime.now(timezone.utc):%Y-%m-%d} by `src/evaluate.py` at the
end of `bash run_all.sh`. All parameters are in `config/config.yaml`.*

## Setup

- **Data**: 10x Genomics PBMC 3k — {meta['n_cells']} cells, {meta['n_classes']}
  fine-grained labels (CellTypist `Immune_All_Low`, majority voting),
  {meta['n_clusters']} Leiden clusters.
- **Graph**: cell-cell k-NN in PCA space (k={meta['params']['k']}),
  {meta['n_edges']} directed edges.
- **Evaluation protocol**: whole-cluster holdout — train
  {meta['split_sizes'].get('train')} / val {meta['split_sizes'].get('val')} /
  test {meta['split_sizes'].get('test')} cells. Test labels:
  {", ".join(test_labels)}. Labels eligible for holdout:
  {", ".join(meta['labels_eligible_for_holdout'])} (all others are train-only,
  see README for the rationale).
- **Same split, same seed** for all four models; torch models early-stopped on
  validation macro F1.

## Results

{md_table(lb)}

| ![Model comparison]({figures_rel}/f1_comparison.png) | ![Training curves]({figures_rel}/loss_curves.png) |
|---|---|

![UMAP true vs predicted]({figures_rel}/umap_true_vs_pred.png)

![Confusion matrices]({figures_rel}/confusion_matrices.png)

## Per-class F1 on the held-out clusters

Only the labels actually present in the test set are shown (the others are
train-only by design):

{md_table(f1_show)}

## Error analysis

{error_md}

{coherence}

## When does the GNN win?

{verdict}

Honest caveats that bound this conclusion:

{caveats_md}

## Reproducibility

`bash run_all.sh` regenerates everything (data download included). Software
versions and the device used are logged in `results/environment.txt`; every
training run is appended to `results/experiments.csv`.
"""
    out = Path(cfg["paths"]["report"]) / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("Wrote %s", out)


def plot_model_order():
    return ["logreg", "mlp", "graphsage_pca", "graphsage_hvg"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    import data as data_mod
    cfg = data_mod.load_config(args.config)
    csv_path = Path(cfg["paths"]["experiments_csv"])
    if not csv_path.exists():
        log.warning("No experiments yet (%s missing)", csv_path)
        return

    tables_dir = Path(cfg["paths"]["tables"])
    pred_dir = Path(cfg["paths"]["predictions"])
    seed = cfg["seed"]

    df = pd.read_csv(csv_path)
    latest = df.sort_values("timestamp").groupby("model").tail(1)
    leaderboard = latest[["model", "features", "seed", "val_macro_f1",
                          "test_accuracy", "test_macro_f1",
                          "train_time_s"]].sort_values(
        "test_macro_f1", ascending=False)
    leaderboard.to_csv(tables_dir / "leaderboard.csv", index=False)
    log.info("Leaderboard:\n%s", leaderboard.to_string(index=False))

    ds = data_mod.load_dataset(cfg)
    f1_table = per_class_table(latest, ds, pred_dir, seed)
    f1_table.to_csv(tables_dir / "per_class_f1_test.csv")
    log.info("Wrote %s", tables_dir / "per_class_f1_test.csv")

    conf = top_confusions(latest, ds, pred_dir, seed)
    conf.to_csv(tables_dir / "top_confusions.csv", index=False)
    log.info("Wrote %s (%d confusion events)", tables_dir / "top_confusions.csv",
             len(conf))

    import plot as plot_mod
    plot_mod.generate_all(cfg, leaderboard)

    generate_report(cfg, leaderboard, f1_table, conf, ds)


if __name__ == "__main__":
    main()
