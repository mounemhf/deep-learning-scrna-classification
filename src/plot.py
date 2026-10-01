#!/usr/bin/env python
"""All study figures (300 dpi, publication styling).

generate_all(cfg) is called by evaluate.py after the metrics tables are
written; plot.py can also be run standalone to regenerate the figures
from the saved predictions and training histories.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import data as data_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sns.set_theme(style="ticks", context="paper")

MODEL_ORDER = ["logreg", "mlp", "graphsage_pca", "graphsage_hvg"]


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote figure %s", path)


def plot_loss_curves(tables_dir: Path, figures_dir: Path, dpi: int) -> None:
    """Train/val loss per torch model, from history_*.csv files."""
    histories = sorted(tables_dir.glob("history_*.csv"))
    if not histories:
        log.warning("No training histories found, skipping loss curves")
        return
    fig, axes = plt.subplots(1, len(histories),
                             figsize=(4.2 * len(histories), 3.4))
    for ax, f in zip(np.atleast_1d(axes), histories):
        h = pd.read_csv(f)
        name = f.stem.replace("history_", "").rsplit("_seed", 1)[0]
        ax.plot(h["epoch"], h["train_loss"], label="train")
        ax.plot(h["epoch"], h["val_loss"], label="validation")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.legend(frameon=False)
    fig.suptitle("Training curves (early stopping on validation macro F1)")
    _save(fig, figures_dir / "loss_curves.png", dpi)


def plot_confusions(confusions: dict, classes: list, figures_dir: Path,
                    dpi: int) -> None:
    """Normalised test confusion matrices of all models, 2x2 panel."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    short = [c if len(c) <= 18 else c[:16] + "…" for c in classes]
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        cm = confusions[name]
        sns.heatmap(cm, ax=ax, xticklabels=short, yticklabels=short,
                    cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={"label": "fraction of true class"})
        ax.set_title(name)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Normalised confusion matrices (test set)")
    fig.tight_layout()
    _save(fig, figures_dir / "confusion_matrices.png", dpi)


def plot_f1_barplot(leaderboard: pd.DataFrame, figures_dir: Path,
                    dpi: int) -> None:
    """Grouped barplot: test accuracy and macro F1 for the four models."""
    df = leaderboard.copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER,
                                 ordered=True)
    df = df.sort_values("model")
    long = df.melt(id_vars="model",
                   value_vars=["test_accuracy", "test_macro_f1"],
                   var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(long, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("test score")
    ax.legend(frameon=False, title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    fig.suptitle("Model comparison on the held-out clusters")
    _save(fig, figures_dir / "f1_comparison.png", dpi)


def plot_umap_true_vs_pred(ds: dict, y_pred: np.ndarray, model_name: str,
                           figures_dir: Path, dpi: int) -> None:
    """UMAP coloured by true vs predicted label; test errors outlined."""
    classes = ds["classes"]
    palette = dict(zip(classes, sns.color_palette("tab10", len(classes))))
    umap = ds["umap"]
    test = ds["test_mask"]
    pred_labels = np.array(classes)[y_pred]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, labels, title in (
            (axes[0], ds["cell_type"], "True labels (CellTypist)"),
            (axes[1], pred_labels, f"Predicted by {model_name}")):
        for c in classes:
            m = labels == c
            ax.scatter(umap[m, 0], umap[m, 1], s=6, color=palette[c],
                       label=c, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    # Outline misclassified test cells on the prediction panel
    errors = test & (pred_labels != ds["cell_type"])
    axes[1].scatter(umap[errors, 0], umap[errors, 1], s=22,
                    facecolors="none", edgecolors="black", linewidths=0.7,
                    label=f"test errors (n={errors.sum()})")
    axes[1].legend(frameon=False, fontsize=7, markerscale=2,
                   loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Cell-type classification on held-out clusters")
    fig.tight_layout()
    _save(fig, figures_dir / "umap_true_vs_pred.png", dpi)


def generate_all(cfg: dict, leaderboard: pd.DataFrame) -> None:
    """Produce every figure from saved artefacts."""
    tables_dir = Path(cfg["paths"]["tables"])
    figures_dir = Path(cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = 300

    ds = data_mod.load_dataset(cfg)
    pred_dir = Path(cfg["paths"]["predictions"])

    confusions, preds = {}, {}
    for name in MODEL_ORDER:
        f = pred_dir / f"{name}_seed{cfg['seed']}.npz"
        if not f.exists():
            log.warning("Missing predictions for %s, skipped in figures", name)
            continue
        z = np.load(f, allow_pickle=False)
        preds[name] = z["y_pred"]
        m = ds["test_mask"]
        cm = pd.crosstab(pd.Series(ds["y"][m], name="true"),
                         pd.Series(z["y_pred"][m], name="pred"))
        cm = cm.reindex(index=range(len(ds["classes"])),
                        columns=range(len(ds["classes"])), fill_value=0)
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        confusions[name] = cm.to_numpy()

    plot_loss_curves(tables_dir, figures_dir, dpi)
    if len(confusions) == len(MODEL_ORDER):
        plot_confusions(confusions, ds["classes"], figures_dir, dpi)
    plot_f1_barplot(leaderboard, figures_dir, dpi)

    # UMAP of the validation-selected best model
    best = leaderboard.sort_values("val_macro_f1", ascending=False).iloc[0]
    if best["model"] in preds:
        plot_umap_true_vs_pred(ds, preds[best["model"]], best["model"],
                               figures_dir, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()
    cfg = data_mod.load_config(args.config)
    leaderboard = pd.read_csv(Path(cfg["paths"]["tables"]) / "leaderboard.csv")
    generate_all(cfg, leaderboard)


if __name__ == "__main__":
    main()
