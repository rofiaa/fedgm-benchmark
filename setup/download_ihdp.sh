#!/bin/sh
# IHDP semi-synthetic benchmark (Hill 2011), the 100 replications of Shalit, Johansson
# & Sontag (2017), as distributed by F. Johansson. Saved to data/ihdp100/ (not tracked).
# Loaded by src/dgps/ihdp.py.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/ihdp100
for s in train test; do
  curl -sfL -o "data/ihdp100/ihdp_npci_1-100.$s.npz" \
    "https://www.fredjo.com/files/ihdp_npci_1-100.$s.npz"
done
cd data/ihdp100
shasum -a 256 -c <<SUMS
750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240  ihdp_npci_1-100.train.npz
a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c  ihdp_npci_1-100.test.npz
SUMS
