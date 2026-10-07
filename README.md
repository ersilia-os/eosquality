![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Quality of Ersilia predictions

Quality scoring for [Ersilia Model Hub](https://ersilia.io) predictions. `eosquality` tries to quantify, via multiple metrics, whether a given run output from Ersilia is "trustworthy". It does **not** estimate the probability that a prediction is correct. Per-column normalisation is done with [`eosframes`](https://github.com/ersilia-os/eosframes).

## Installation

Install the latest version of `eosquality` directly from GitHub:

```bash
pip install git+https://github.com/ersilia-os/eosquality.git
```

The CLI is then available as `eosquality`. Start by setting it up, which fetches the reference library and its indices:

```bash
eosquality setup
```

This will take a while. The library is stored under `~/.eosquality/`.

## Quick start

The typical workflow is two commands: `fit` once per Ersilia model, then `run` on any query dataset against the saved artifacts. `fit` takes the model's predictions on the reference library (`-r`), its per-output-column training sets (`-t`), or both, and fits the matching scores. File and folder names carry the model id and version (`[prefix_]<eos_id>_<version>`, e.g. `reference_eos4e40_v1.csv`); a name without them is an error. See the [CLI docs](docs/cli.md).

### Fitting a reference library

The input CSV must hold the model's predictions on the **exact** molecules of the canonical Ersilia reference library for your installed `eosquality` version, in library order. It needs a `key` column, an `input` (SMILES) column, and one numeric column per model output. The artifacts folder (`-a`) stores the fitted scores.

There is one and only one reference library per major version of `eosquality`, so the molecule set is fixed by your install. If the SMILES entries in the input CSV don't match that library, `fit` refuses with an error.

```bash
eosquality fit -r reference_eos4e40_v1.csv -a artifacts_eos4e40_v1/
# optionally with training sets: -t training_eos4e40_v1/
# skip a score: --exclude ref_signal
```

Please check [Isaura](https://github.com/ersilia-os/isaura) for a large store of pre-calculations across Ersilia models.

### Fitting training sets

`-t` takes a folder with one `<output_column>.csv` per model output, each
with a `smiles` column. Given both `-r` and `-t`, the scores cover only the
output columns that have a training set.

```bash
eosquality fit -r reference_eos4e40_v1.csv -t training_eos4e40_v1/ -a artifacts_eos4e40_v1/
```

### Running against new samples

At querying time, `run` loads a fitted artifacts folder and scores any query CSV containing Ersilia results for the same model. A training-only artifact needs nothing but SMILES, in an `input` or `smiles` column. The output CSV has `key`, `input`, and, for each score, a calibrated column in `(0, 1]` plus its `*_raw` value, named `ref_<score>` for the reference scores and `trn_<score>` for the training scores.

```bash
eosquality run -i query_eos4e40_v1.csv -a artifacts_eos4e40_v1/ -o quality_eos4e40_v1.csv
```

## Scores

Reference-modality scores compare a query against the model's own predictions on the reference library, which is **not** ground truth. Each is calibrated so that reference molecules score roughly Uniform(0, 1). Training-modality scores compare it against the model's training sets instead, calibrated so that training molecules score roughly Uniform(0, 1).

| Score (column) | Question |
|---|---|
| **Typicality** (`ref_typicality`) | Are the predicted values ones the model commonly produces? |
| **Extremity** (`ref_extremity`) | Are the predicted values far from the centre of the model's output range? |
| **Support** (`ref_support`) | Does the reference library contain a close analogue of the molecule? |
| **Consistency** (`ref_consistency`) | Do the predictions agree with those for chemically similar reference molecules, given how similar they are? |
| **Signal** (`ref_signal`, provisional) | Is the prediction driven by a few chemical descriptors? |
| **Training similarity** (`trn_tanimoto`) | How far is the molecule from the model's training molecules, compared with how close they are to each other? Raw and as a percentile of the training set's own distances. |
| **Training physchem** (`trn_physchem`) | The same question in physicochemical descriptor space. |
| **Training match** (`trn_match`, `trn_scaffold`) | Is the same structure, or the same Murcko scaffold, in a training set? 1 or 0. |

## Documentation

- [Concepts](docs/concepts.md): how each score is computed and calibrated
- [CLI](docs/cli.md) and [Python API](docs/api.md)
- [Architecture](docs/diagram.md): fit/run data flow and save layout
- [Reference library](docs/reference-library.md): versioning, compatibility, maintainer release steps
- [Project status](docs/status.md): current results, known limitations, open questions

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
