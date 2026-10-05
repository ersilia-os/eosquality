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

The typical workflow is two commands: `fit` once per Ersilia model, then `run` on any query dataset against the saved artifacts. `fit` takes the model's predictions on the reference library (`--reference`), its per-output-column training sets (`--training-sets`), or both, and fits the matching scores. See the [CLI docs](docs/cli.md).

### Fitting a reference library

The input CSV must hold the model's predictions on the **exact** molecules of the canonical Ersilia reference library for your installed `eosquality` version, in library order. It needs a `key` column, an `input` (SMILES) column, and one numeric column per model output. The filename encodes the model and version (e.g. `eos4e40_v1.csv`). The output folder stores the fitted artifacts.

There is one and only one reference library per major version of `eosquality`, so the molecule set is fixed by your install. If the SMILES entries in the input CSV don't match that library, `fit` refuses with an error.

```bash
eosquality fit --reference reference_eos4e40_v1.csv --output artifacts_eos4e40_v1/
# optionally with training sets: --training-sets training_eos4e40_v1/
```

Please check [Isaura](https://github.com/ersilia-os/isaura) for a large store of pre-calculations across Ersilia models.

### Running against new samples

At querying time, `run` loads a fitted artifacts folder and scores any query CSV containing Ersilia results for the same model. The output CSV has `key`, `input`, and, for each score, a calibrated column in `(0, 1]` plus its `*_raw` value.

```bash
eosquality run --input query_eos4e40_v1.csv --artifacts artifacts_eos4e40_v1/ --output quality_eos4e40_v1.csv
```

## Scores

Reference-modality scores compare a query against the model's own predictions on the reference library, which is **not** ground truth. Each is calibrated so that reference molecules score roughly Uniform(0, 1).

| Score | Question |
|---|---|
| **Typicality** | Are the predicted values ones the model commonly produces? |
| **Extremity** | Are the predicted values far from the centre of the model's output range? |
| **Support** | Does the reference library contain a close analogue of the molecule? |
| **Consistency** | Do the predictions agree with those for chemically similar reference molecules, given how similar they are? |
| **Signal** *(opt-in)* | Is the prediction driven by a few chemical descriptors? |
| **Training distance** *(with `--training-sets`)* | How far is the molecule from the model's training molecules, compared with how close they are to each other? One value for the whole model. |
| **Training difficulty** *(training sets with labels)* | How hard is the molecule to predict, judging by where a learned error model finds the training data hard? One value for the whole model. |

## Documentation

- [Concepts](docs/concepts.md): how each score is computed and calibrated
- [CLI](docs/cli.md) and [Python API](docs/api.md)
- [Architecture](docs/diagram.md): fit/run data flow and save layout
- [Reference library](docs/reference-library.md): versioning, compatibility, maintainer release steps
- [Project status](docs/status.md): current results, known limitations, open questions

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
