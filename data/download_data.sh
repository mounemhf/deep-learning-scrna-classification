#!/usr/bin/env bash
# Download the 10x Genomics PBMC 3k filtered gene-barcode matrix (~7 MB
# archive). This is the same public dataset as the companion pipeline
# project; read-level QC is out of scope here, only the matrix is needed.
#
# Usage:
#   bash data/download_data.sh           # download (skip if present)
#   bash data/download_data.sh --force   # re-download

set -euo pipefail

MATRIX_URL="https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
MATRIX_DIR="${RAW_DIR}/filtered_gene_bc_matrices/hg19"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -f "${MATRIX_DIR}/matrix.mtx" && "${FORCE}" -eq 0 ]]; then
    echo "[skip] filtered matrix already present in ${MATRIX_DIR}"
    exit 0
fi

mkdir -p "${RAW_DIR}"
archive="${RAW_DIR}/pbmc3k_filtered_gene_bc_matrices.tar.gz"
echo "[download] ${MATRIX_URL}"
curl -fSL --retry 3 --retry-delay 5 -o "${archive}" "${MATRIX_URL}"
tar -xzf "${archive}" -C "${RAW_DIR}"
rm -f "${archive}"

for f in matrix.mtx barcodes.tsv genes.tsv; do
    if [[ ! -f "${MATRIX_DIR}/${f}" ]]; then
        echo "[error] expected file missing after extraction: ${MATRIX_DIR}/${f}" >&2
        exit 1
    fi
done
echo "[ok] filtered matrix ready in ${MATRIX_DIR}"
