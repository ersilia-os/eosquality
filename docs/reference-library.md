# Reference library

## Identity and versioning

Each major version of `eosquality` ships exactly one reference library, `ersilia_reference_library_vN` (`LIBRARY_ID` in `src/eosquality/library/identity.py`). Its molecules are the [canonical Ersilia reference library](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs).

The same name is used everywhere:
- the `library_name` field in the library folder's `metadata.json`
- the source CSV stem under `data/libraries/`
- the library folder under `data/indices/`
- the user-cache folder
- the S3 path segment

The library's `vN` must equal the package's major version. Importing a release where they disagree raises an error.

**When to bump.** Any change that alters scores needs a new library, and therefore a new package major version. That covers adding or removing molecules, correcting SMILES, and anything that changes the connectivity keys. Edits to metadata only do not.

**Status.** The package is currently `0.1.0` with library `ersilia_reference_library_v0` (about 1.36M molecules).

## Artifact compatibility

Saved artifacts record the following in `reference_mode/shared/metadata.json`:
- `library_id`
- `eosquality_version`
- `format_version` (the on-disk layout and score semantics; currently 11)
- `library_path` (for custom libraries)

The training modality is versioned separately: `training_mode/training_sets/metadata.json` holds `training_format_version` (currently 11). Adding or changing the training modality therefore never invalidates reference artifacts.

`ErsiliaQuality.load` rejects artifacts in these cases:
- the format version is different, or the folder uses the old flat layout (no `reference_mode/` / `training_mode/`) → `ArtifactVersionError`; refit;
- the library is not this install's canonical library and no custom library path is recorded → `IncompatibleArtifactsError`;
- the package major version is different → `IncompatibleArtifactsError`.

When an artifact is fit on the canonical library, only its identity is stored, not a path, so the artifact is portable between machines. The library is resolved again at run time (see [cli.md](cli.md#the-reference-library)).

When an artifact is fit against a custom library (`library=` in the Python API), the absolute path of that library folder is stored, and the folder must still exist when `run` is called: `ref_match` reads its keys from it. A typicality/extremity-only artifact does not need the library at run time.

## Library folder contents

| file | content |
|---|---|
| `smiles.csv` | the library SMILES, in order; a model's reference predictions must be for exactly these molecules, in this order |
| `metadata.json` | `library_name` (the identity), `n_samples`, number of keys, `rdkit_version` (the RDKit the keys were built with), eosquality version, build timestamp |
| `connectivity_keys.npz` | the sorted unique InChIKey connectivity layers of the (standardised) molecules and of their Murcko scaffolds, for `ref_match` / `ref_scaffold` |

**Version pinning.** The connectivity keys depend on the RDKit version (InChI generation and scaffold perception can change between releases). The build records the RDKit version in `metadata.json`, and reading the keys (at `fit` and at `run`) raises `IncompatibleArtifactsError` if the installed RDKit differs, rather than returning keys that may disagree. Install the recorded RDKit, rebuild the library, or exclude `ref_match`. Libraries built before the version was recorded are not checked.

## Releasing a new library (maintainers)

1. Bump `LIBRARY_ID` and the package major version together.
2. Build the library folder from the new Ersilia SMILES CSV:
   ```bash
   eosquality build --input data/libraries/ersilia_reference_library_v1.csv \
                    --output data/indices/ersilia_reference_library_v1/
   ```
   The connectivity keys take a few minutes on several cores for the 1.35M molecules.
3. Upload the CSV and the folder with [eosvc](https://github.com/ersilia-os/eosvc). The repo's `access.json` routes `data/` to the public bucket:
   ```bash
   eosvc upload --path data/libraries/ersilia_reference_library_v1.csv
   eosvc upload --path data/indices/ersilia_reference_library_v1/
   ```
