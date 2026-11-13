#!/bin/sh
# International Stroke Trial database, version 2 (Sandercock, Niewada & Czlonkowska 2011,
# https://doi.org/10.7488/ds/104, ODC-By 1.0), from Edinburgh DataShare.
# Saved to data/ist/ (not tracked): data, variable list, data paper (Trials 2011) and the
# version-2 commentary.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/ist
B=https://datashare.ed.ac.uk/server/api/core/bitstreams
curl -sfL -o data/ist/IST_corrected.csv     "$B/eca52b56-6d1b-47b3-814a-0b7089f8c870/content"
curl -sfL -o data/ist/IST_variables.csv     "$B/069dc665-f3c8-4c95-8ec0-bf6736e0c8ac/content"
curl -sfL -o data/ist/IST_trials2011.pdf    "$B/095fd50c-8c11-4ba0-8ae2-c638dcbd04ba/content"
curl -sfL -o data/ist/IST_v2_commentary.pdf "$B/20442eba-5305-4d00-9b21-1078081a0863/content"
cd data/ist
shasum -a 256 -c <<SUMS
a350aed2329bbcde23c05caaa96b9275b44b017adecc2fd02231d590454695d5  IST_corrected.csv
bab564a5e44328d4c8c1cce534d029366d2847c79548d8a5de5f00f2d1bbcf8c  IST_variables.csv
SUMS
