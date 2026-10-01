# Results

Everything in this directory is **generated** by `bash run_all.sh`.
Only the final figures (`figures/*.png`) are tracked by git, as the
portfolio showcase.

Expected layout after a full run:

```
results/
├── environment.txt     # pinned software versions + device actually used
├── experiments.csv     # one row per training run (append-only)
├── best_model.pt       # best model by validation macro F1
├── processed/          # prepared AnnData, graph.npz, splits.npz
├── predictions/        # per-cell predictions per model (npz)
├── tables/             # split_summary, leaderboard, per-class metrics
├── figures/            # 300 dpi PNGs used in the report
└── logs/               # one log per run_all.sh step
```
