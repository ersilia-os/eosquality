#!/usr/bin/env bash
# Manual end-to-end smoke check across every model with shipped example data.
# Invocable from any cwd — the script jumps to the repo root so the package's
# library auto-resolution finds data/libraries/ and data/indices/ locally
# (otherwise it tries an S3 download).
#
# For each model in MODELS, this does:
#   1. eosquality fit  — emh_paper_<eos>_v1.csv → outputs/artifacts/
#   2. eosquality run  — molecules_1000_<eos>_v1.csv → outputs/scores_molecules_*.csv
#   3. eosquality run  — emh_paper_head_<eos>_v1.csv → outputs/scores_emh_paper_head_*.csv
#   4. clean up outputs/artifacts/ before the next model

set -eu

cd "$(dirname "$0")/.."

MODELS=(eos4e40 eos7m30 eos3804 eos42ez)

mkdir -p outputs

for EOS in "${MODELS[@]}"; do
    echo "==> Starting EOS: $EOS"
    rm "outputs/scores_molecules_1000_${EOS}_v1.csv" "outputs/scores_emh_paper_head_${EOS}_v1.csv" 2>/dev/null || true

    echo "==> [$EOS] eosquality fit..."
    eosquality fit \
        -i "data/fit_examples/emh_paper_${EOS}_v1.csv" \
        -o outputs/artifacts

    echo "==> [$EOS] eosquality run (molecules_1000)..."
    eosquality run \
        -i "data/run_examples/molecules_1000_${EOS}_v1.csv" \
        -a outputs/artifacts \
        -o "outputs/scores_molecules_1000_${EOS}_v1.csv"

    echo "==> [$EOS] eosquality run (emh_paper_head)..."
    eosquality run \
        -i "data/run_examples/emh_paper_head_${EOS}_v1.csv" \
        -a outputs/artifacts \
        -o "outputs/scores_emh_paper_head_${EOS}_v1.csv"

    echo "==> [$EOS] cleaning up artifacts..."
    rm -rf outputs/artifacts
done

echo "All models complete. Score files in outputs/."
