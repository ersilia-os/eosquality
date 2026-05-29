#!/usr/bin/env bash
# Fit + run every model present in data/fit_examples/ against every
# matching 1000-molecule query in data/run_examples/, using *all* quality
# scores (typicality, extremity, support, consistency, signal).
#
# Produces, in outputs/:
#   outputs/artifacts_<eos>/                          per-model fit artifacts
#   outputs/scores_<dataset>_<eos>_v1.csv             per-(model, dataset) scores
#
# Auto-discovers both the model list and the per-model query list — no
# hardcoded MODELS / DATASETS arrays. To skip a model, remove its fit
# corpus from data/fit_examples/. To skip a dataset for a model, remove
# the corresponding *_1000_<eos>_v1.csv file from data/run_examples/.

set -eu

cd "$(dirname "$0")/.."

ALL_SCORES="typicality,extremity,support,consistency,signal"

mkdir -p outputs

shopt -s nullglob
FIT_FILES=(data/fit_examples/emh_paper_*_v1.csv)
shopt -u nullglob

if [ ${#FIT_FILES[@]} -eq 0 ]; then
    echo "no fit corpora found under data/fit_examples/emh_paper_*_v1.csv" >&2
    exit 1
fi

for FIT_INPUT in "${FIT_FILES[@]}"; do
    BASENAME="${FIT_INPUT##*/}"           # emh_paper_eos4e40_v1.csv
    STRIPPED="${BASENAME#emh_paper_}"     # eos4e40_v1.csv
    EOS="${STRIPPED%_v1.csv}"             # eos4e40

    ARTIFACTS="outputs/artifacts_${EOS}"
    echo "==> [$EOS] eosquality fit (all scores) → $ARTIFACTS"
    rm -rf "$ARTIFACTS"
    eosquality fit \
        -i "$FIT_INPUT" \
        -o "$ARTIFACTS" \
        --scores "$ALL_SCORES"

    shopt -s nullglob
    QUERY_FILES=(data/run_examples/*_1000_"${EOS}"_v1.csv)
    shopt -u nullglob

    if [ ${#QUERY_FILES[@]} -eq 0 ]; then
        echo "==> [$EOS] no 1000-mol queries at data/run_examples/*_1000_${EOS}_v1.csv"
        continue
    fi

    for QUERY_INPUT in "${QUERY_FILES[@]}"; do
        QBASE="${QUERY_INPUT##*/}"                  # drugs_1000_eos4e40_v1.csv
        DATASET="${QBASE%_${EOS}_v1.csv}"           # drugs_1000
        OUTPUT="outputs/scores_${DATASET}_${EOS}_v1.csv"

        echo "==> [$EOS] eosquality run ($DATASET) → $OUTPUT"
        eosquality run \
            -i "$QUERY_INPUT" \
            -a "$ARTIFACTS" \
            -o "$OUTPUT"
    done
done

echo "==> done. score CSVs in outputs/."
