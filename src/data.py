#!/usr/bin/env python
"""Dataset loading, feature sets and split handling for the study.

data.py is the single entry point used by train.py / evaluate.py:
it reads the artefacts produced by prepare_data.py and exposes feature
matrices (50 PCs or top-2000 HVG expression), integer labels, boolean
train/val/test masks and the k-NN graph as NumPy arrays, plus a PyG
converter for the graph models.
"""

import logging
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml

log = logging.getLogger(__name__)

FEATURE_SETS = ("pca", "hvg")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict):
    """Return the torch device, honouring config `device: auto`."""
    import torch

    want = cfg.get("device", "auto")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(want)
    log.info("Using device: %s", device)
    return device


def load_dataset(cfg: dict) -> dict:
    """Load prepared data as plain NumPy arrays (framework-agnostic)."""
    processed = Path(cfg["paths"]["processed"])
    adata = sc.read_h5ad(processed / "pbmc3k_prepared.h5ad")
    splits = np.load(processed / "splits.npz")
    graph = np.load(processed / "graph.npz")

    ds = {
        "x_pca": np.asarray(adata.obsm["X_pca"], dtype=np.float32),
        "x_hvg": np.asarray(adata.obsm["X_hvg"], dtype=np.float32),
        "y": splits["y"].astype(np.int64),
        "classes": [str(c) for c in splits["classes"]],
        "train_mask": splits["train"].astype(bool),
        "val_mask": splits["val"].astype(bool),
        "test_mask": splits["test"].astype(bool),
        "edge_index": graph["edge_index"].astype(np.int64),
        "umap": np.asarray(adata.obsm["X_umap"], dtype=np.float32),
        "cell_type": adata.obs["cell_type"].to_numpy(),
    }
    n_train = ds["train_mask"].sum()
    log.info("Dataset: %d cells, %d classes, %d edges (train %d / val %d / test %d)",
             len(ds["y"]), len(ds["classes"]), ds["edge_index"].shape[1],
             n_train, ds["val_mask"].sum(), ds["test_mask"].sum())
    return ds


def feature_matrix(ds: dict, feature_set: str) -> np.ndarray:
    if feature_set == "pca":
        return ds["x_pca"]
    if feature_set == "hvg":
        return ds["x_hvg"]
    raise ValueError(f"unknown feature set: {feature_set!r} "
                     f"(expected one of {FEATURE_SETS})")


def to_pyg(ds: dict, feature_set: str, device):
    """Build a torch_geometric Data object with masks."""
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.from_numpy(feature_matrix(ds, feature_set)),
        edge_index=torch.from_numpy(ds["edge_index"]),
        y=torch.from_numpy(ds["y"]),
        train_mask=torch.from_numpy(ds["train_mask"]),
        val_mask=torch.from_numpy(ds["val_mask"]),
        test_mask=torch.from_numpy(ds["test_mask"]),
    )
    return data.to(device)
