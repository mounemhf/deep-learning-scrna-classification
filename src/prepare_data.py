#!/usr/bin/env python
"""Prepare the PBMC 3k dataset for the comparative classification study.

Steps: QC filtering, normalisation, highly variable genes, PCA, Leiden
clustering, CellTypist labels (Immune_All_Low, majority voting), a
whole-cluster holdout train/val/test split, and the cell-cell k-NN graph
in PCA space. Saves processed.h5ad, graph/split arrays, summary tables
and a pinned-software report (results/environment.txt).
"""

import argparse
import importlib.metadata
import json
import logging
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KEY_PACKAGES = ["torch", "torch_geometric", "scikit-learn", "scanpy",
                "celltypist", "leidenalg", "anndata", "numpy", "pandas",
                "matplotlib", "seaborn"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_knn_graph(pca: np.ndarray, k: int) -> np.ndarray:
    """k-NN graph in PCA space, symmetrised by union, no self-loops."""
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(pca).kneighbors(pca)
    n = pca.shape[0]
    edges = set()
    for i in range(n):
        for j in idx[i]:
            if i != j:
                edges.add((i, j))
                edges.add((j, i))
    edge_index = np.array(sorted(edges), dtype=np.int64).T
    log.info("k-NN graph: %d nodes, %d directed edges (k=%d)", n, edge_index.shape[1], k)
    return edge_index


def assign_splits(obs: pd.DataFrame, val_frac: float, test_frac: float,
                  seed: int):
    """Whole-cluster holdout split.

    The largest cluster of every label is first reserved for train (the
    label is thus guaranteed to be seen in training). Remaining clusters
    of multi-cluster labels are greedily assigned to whichever of
    val/test is furthest below its cell target and does not yet contain
    that label. Single-cluster labels stay entirely in train.
    """
    cl_label = obs.groupby("leiden")["cell_type"].agg(
        lambda s: s.value_counts().idxmax())
    cl_size = obs["leiden"].value_counts()
    label2clusters = {}
    for cl, lab in cl_label.items():
        label2clusters.setdefault(lab, []).append(cl)
    eligible = {lab for lab, cls in label2clusters.items() if len(cls) >= 2}

    n = len(obs)
    targets = {"val": val_frac * n, "test": test_frac * n}
    sizes = {"train": 0, "val": 0, "test": 0}
    labels_in = {"val": set(), "test": set()}
    assignment = {}

    # Reserve the largest cluster of each label for train
    reserved = set()
    for lab, cls in label2clusters.items():
        largest = max(cls, key=lambda c: cl_size[c])
        assignment[largest] = "train"
        reserved.add(largest)
        sizes["train"] += cl_size[largest]

    # Greedy holdout assignment for the remaining clusters, largest first:
    # the holdout budget is limited by the number of eligible LABELS (a label
    # appears at most once per split), so larger clusters make the held-out
    # sets statistically sturdier at equal label diversity
    rest = [c for c in cl_size.sort_values(ascending=False, kind="stable").index
            if c not in reserved]
    for cl in rest:
        lab = cl_label[cl]
        placed = False
        if lab in eligible:
            by_deficit = sorted(("test", "val"),
                                key=lambda s: targets[s] - sizes[s],
                                reverse=True)
            for s in by_deficit:
                if lab not in labels_in[s] and sizes[s] < targets[s]:
                    assignment[cl] = s
                    sizes[s] += cl_size[cl]
                    labels_in[s].add(lab)
                    placed = True
                    break
        if not placed:
            assignment[cl] = "train"
            sizes["train"] += cl_size[cl]

    split = obs["leiden"].map(assignment)

    # Fail loudly rather than logging NaN metrics on empty splits
    for s in ("val", "test"):
        n_s = int((split == s).sum())
        labels_s = set(obs.loc[split == s, "cell_type"])
        assert n_s > 0, f"empty {s} split: raise clustering_resolution or " \
                        f"lower the {s} fraction"
        assert labels_s <= set(obs.loc[split == "train", "cell_type"]), \
            f"split constraint violated for {s}: {labels_s} not all in train"
        log.info("%s: %d cells, labels %s", s, n_s, sorted(labels_s))
    return split, assignment, eligible


def write_environment(path: Path) -> None:
    lines = [
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
    ]
    for pkg in KEY_PACKAGES:
        try:
            lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{pkg}: NOT INSTALLED")
    try:
        import torch
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
    except ImportError:
        lines.append("cuda_available: torch not installed")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", path)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pre = cfg["prepare"]
    paths = cfg["paths"]
    set_seeds(cfg["seed"])

    out_processed = Path(paths["processed"])
    out_tables = Path(paths["tables"])
    for d in (out_processed, out_tables, Path(paths["logs"]),
              Path(paths["figures"]), Path(paths["predictions"])):
        d.mkdir(parents=True, exist_ok=True)

    log.info("Loading 10x matrix from %s", paths["matrix_dir"])
    adata = sc.read_10x_mtx(paths["matrix_dir"], var_names="gene_symbols",
                            cache=False)
    adata.var_names_make_unique()
    n_raw = adata.n_obs

    # QC filtering (same thresholds as the companion pipeline project)
    adata.var["mt"] = adata.var_names.str.startswith(pre["mito_prefix"])
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=pre["min_genes"])
    sc.pp.filter_genes(adata, min_cells=pre["min_cells"])
    adata = adata[adata.obs["pct_counts_mt"] < pre["max_pct_mito"]].copy()
    adata = adata[adata.obs["total_counts"] < pre["max_counts"]].copy()
    log.info("QC: %d -> %d cells", n_raw, adata.n_obs)

    # Normalisation, HVG, PCA (50 PCs are a feature set for the models)
    sc.pp.normalize_total(adata, target_sum=pre["target_sum"])
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=pre["n_top_hvg"],
                                flavor="seurat")
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=pre["n_pcs"], mask_var="highly_variable",
              svd_solver="arpack", random_state=cfg["seed"])
    adata.obsm["X_hvg"] = np.asarray(
        adata[:, adata.var["highly_variable"]].X, dtype=np.float32)

    # Leiden clusters (high resolution: fine labels should span clusters)
    sc.pp.neighbors(adata, n_neighbors=pre["n_neighbors"], n_pcs=pre["n_pcs"],
                    random_state=cfg["seed"])
    sc.tl.leiden(adata, resolution=pre["clustering_resolution"],
                 random_state=cfg["seed"], flavor="leidenalg",
                 key_added="leiden")

    # CellTypist fine-grained labels, majority voting over Leiden clusters
    import celltypist
    log.info("Annotating with CellTypist %s", pre["celltypist_model"])
    predictions = celltypist.annotate(
        adata.raw.to_adata(), model=pre["celltypist_model"],
        majority_voting=True, over_clustering="leiden")
    adata.obs["predicted_labels"] = predictions.predicted_labels["predicted_labels"]
    adata.obs["cell_type"] = predictions.predicted_labels["majority_voting"]
    adata.obs["conf_score"] = predictions.probability_matrix.max(axis=1).values

    # Drop cells whose label is too rare to classify meaningfully
    counts = adata.obs["cell_type"].value_counts()
    keep_labels = counts[counts >= pre["min_class_cells"]].index
    adata = adata[adata.obs["cell_type"].isin(keep_labels)].copy()
    log.info("Kept %d labels with >= %d cells: %s",
             len(keep_labels), pre["min_class_cells"], list(keep_labels))

    # Renumber clusters contiguously after cell filtering
    remap = {old: str(i) for i, old in
             enumerate(sorted(adata.obs["leiden"].unique(), key=int))}
    adata.obs["leiden"] = adata.obs["leiden"].map(remap).astype("category")

    sc.tl.umap(adata, random_state=cfg["seed"])

    # Whole-cluster holdout split
    split, assignment, eligible = assign_splits(
        adata.obs, cfg["split"]["val_fraction"], cfg["split"]["test_fraction"],
        cfg["seed"])
    adata.obs["split"] = split.values
    log.info("Split sizes: %s", split.value_counts().to_dict())

    # k-NN graph in PCA space
    edge_index = build_knn_graph(np.asarray(adata.obsm["X_pca"]),
                                 cfg["graph"]["k"])

    # Class encoding
    classes = np.array(sorted(adata.obs["cell_type"].unique()))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = adata.obs["cell_type"].map(class_to_int).to_numpy(dtype=np.int64)

    # --- Persist everything ---
    adata.write_h5ad(out_processed / "pbmc3k_prepared.h5ad")
    np.savez(out_processed / "graph.npz", edge_index=edge_index)
    np.savez(out_processed / "splits.npz",
             train=(split == "train").to_numpy(),
             val=(split == "val").to_numpy(),
             test=(split == "test").to_numpy(),
             y=y, classes=classes)

    summary = pd.crosstab(adata.obs["cell_type"], adata.obs["split"])
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(out_tables / "split_summary.csv")

    meta = {
        "n_cells": int(adata.n_obs),
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "n_clusters": int(adata.obs["leiden"].nunique()),
        "n_edges": int(edge_index.shape[1]),
        "split_sizes": {k: int(v) for k, v in split.value_counts().items()},
        "cluster_assignment": assignment,
        "labels_eligible_for_holdout": sorted(eligible),
        "labels_train_only": sorted(set(classes) - eligible),
        "params": pre | cfg["graph"] | cfg["split"],
        "seed": cfg["seed"],
    }
    (out_processed / "prepare_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_environment(Path("results/environment.txt"))
    log.info("Preparation complete: %d cells, %d classes, %d edges",
             adata.n_obs, len(classes), edge_index.shape[1])


if __name__ == "__main__":
    sys.exit(main())
