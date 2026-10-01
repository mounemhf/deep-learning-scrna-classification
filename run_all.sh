#!/usr/bin/env bash
# Full comparative study, one command. Requires the conda environment:
#   mamba env create -f environment.yml && conda activate dl-scrna
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs

echo "[1/3] Download data"
bash data/download_data.sh 2>&1 | tee results/logs/download.log

echo "[2/3] Prepare data (QC, labels, cluster-holdout split, k-NN graph)"
python src/prepare_data.py --config config/config.yaml 2>&1 | tee results/logs/prepare.log

echo "[3/3] Train and evaluate all models (same split, same seed)"
for model in logreg mlp graphsage_pca graphsage_hvg; do
    echo "  -> ${model}"
    python src/train.py --model "${model}" --config config/config.yaml \
        2>&1 | tee "results/logs/train_${model}.log"
done

python src/evaluate.py --config config/config.yaml 2>&1 | tee results/logs/evaluate.log
echo "[done] see results/experiments.csv and results/tables/leaderboard.csv"
