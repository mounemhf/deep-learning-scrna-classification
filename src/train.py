#!/usr/bin/env python
"""Train one model of the comparative study and log the run.

One generic path drives every model: logistic regression (scikit-learn,
no early stopping — iterative solver) and the torch models (MLP,
GraphSAGE on PCA or HVG features), all full-batch with early stopping
on validation macro F1. Each run appends to results/experiments.csv,
persists per-cell predictions and its training history, and the best
checkpoint by validation macro F1 is saved as results/best_model.pt.
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as data_mod
import evaluate as eval_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logreg":        {"features": "pca", "kind": "sklearn"},
    "mlp":           {"features": "pca", "kind": "torch"},
    "graphsage_pca": {"features": "pca", "kind": "torch"},
    "graphsage_hvg": {"features": "hvg", "kind": "torch"},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="override config seed")
    return p.parse_args()


def train_logreg(ds: dict, seed: int):
    """Baseline 1: multinomial logistic regression on 50 PCs."""
    from sklearn.linear_model import LogisticRegression

    x = data_mod.feature_matrix(ds, "pca")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(x[ds["train_mask"]], ds["y"][ds["train_mask"]])
    train_time = time.perf_counter() - t0
    meta = {"hyperparams": {"max_iter": 1000},
            "train_time_s": round(train_time, 3), "epochs": ""}
    return clf, meta


def train_torch(model, data, cfg: dict):
    """Generic full-batch training loop with early stopping on val macro F1."""
    import torch
    from sklearn.metrics import f1_score

    tr = cfg["training"]
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                           weight_decay=tr["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs = -1.0, None, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = float(loss_fn(out[data.val_mask],
                                     data.y[data.val_mask]).item())
            val_pred = out.argmax(dim=1)
            val_f1 = float(f1_score(
                data.y[data.val_mask].cpu().numpy(),
                val_pred[data.val_mask].cpu().numpy(),
                average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": round(loss.item(), 5),
                        "val_loss": round(val_loss, 5),
                        "val_macro_f1": round(val_f1, 4)})

        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(
                model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= tr["patience"]:
                log.info("Early stopping at epoch %d (best val macro F1 %.4f)",
                         epoch, best_f1)
                break

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    log.info("Trained %d epochs in %.1f s (best val macro F1 %.4f)",
             epoch, train_time, best_f1)
    return model, history, epoch, train_time


def maybe_save_best(cfg: dict, name: str, artifact, val_f1: float,
                    meta: dict) -> None:
    """Save results/best_model.pt if this run beats previous val macro F1."""
    import torch

    csv_path = Path(cfg["paths"]["experiments_csv"])
    best_so_far = -1.0
    if csv_path.exists():
        prev = pd.to_numeric(pd.read_csv(csv_path)["val_macro_f1"],
                             errors="coerce")
        if prev.notna().any():
            best_so_far = float(prev.max())
    if val_f1 > best_so_far:
        torch.save({"model": name, "artifact": artifact, "meta": meta,
                    "val_macro_f1": val_f1}, cfg["paths"]["best_model"])
        log.info("New best model: %s (val macro F1 %.4f) -> %s",
                 name, val_f1, cfg["paths"]["best_model"])


def main():
    args = parse_args()
    cfg = data_mod.load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_mod.set_seeds(seed)
    ds = data_mod.load_dataset(cfg)
    for split in ("train", "val", "test"):
        if not ds[f"{split}_mask"].any():
            raise SystemExit(f"error: empty {split} split — re-run "
                             "src/prepare_data.py first")

    spec = MODEL_REGISTRY[args.model]
    if spec["kind"] == "sklearn":
        model, meta = train_logreg(ds, seed)
        x_all = data_mod.feature_matrix(ds, "pca")
        y_pred = model.predict(x_all)
        artifact = {"sklearn": model}
        history = None
    else:
        import torch
        from models import build_model

        device = data_mod.get_device(cfg)
        data = data_mod.to_pyg(ds, spec["features"], device)
        model = build_model(args.model, data.num_features,
                            len(ds["classes"]), cfg).to(device)
        model, history, epochs, train_time = train_torch(model, data, cfg)
        model.eval()
        with torch.no_grad():
            y_pred = model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()
        meta = {"hyperparams": {"lr": cfg["training"]["lr"],
                                "weight_decay": cfg["training"]["weight_decay"],
                                "hidden_dim": cfg["training"]["hidden_dim"],
                                "dropout": cfg["training"]["dropout"],
                                "max_epochs": cfg["training"]["max_epochs"],
                                "patience": cfg["training"]["patience"]},
                "train_time_s": round(train_time, 3), "epochs": epochs}
        artifact = {"state_dict": model.state_dict(),
                    "in_dim": data.num_features, "n_classes": len(ds["classes"])}

    metrics = {}
    for split in ("val", "test"):
        mask = ds[f"{split}_mask"]
        m = eval_mod.compute_metrics(ds["y"][mask], y_pred[mask],
                                     len(ds["classes"]))
        metrics[f"{split}_accuracy"] = round(m["accuracy"], 4)
        metrics[f"{split}_macro_f1"] = round(m["macro_f1"], 4)
        log.info("[%s] %s: accuracy=%.4f macro F1=%.4f (%d classes evaluated)",
                 args.model, split, m["accuracy"], m["macro_f1"],
                 m["n_eval_classes"])

    # Persist predictions and training history for evaluate.py / plot.py
    pred_dir = Path(cfg["paths"]["predictions"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"{args.model}_seed{seed}.npz",
             y_pred=y_pred, classes=np.array(ds["classes"]))
    if history:
        pd.DataFrame(history).to_csv(
            Path(cfg["paths"]["tables"]) / f"history_{args.model}_seed{seed}.csv",
            index=False)

    maybe_save_best(cfg, args.model, artifact, metrics["val_macro_f1"], meta)
    eval_mod.append_experiment(
        cfg["paths"]["experiments_csv"],
        {"model": args.model, "features": spec["features"], "seed": seed,
         "hyperparams": meta["hyperparams"], "epochs": meta["epochs"],
         "train_time_s": meta["train_time_s"], **metrics})


if __name__ == "__main__":
    main()
