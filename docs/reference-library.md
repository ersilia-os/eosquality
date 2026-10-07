# Reference library

## Identity and versioning

Each major version of `eosquality` ships exactly one reference library, `ersilia_reference_library_vN` (`LIBRARY_ID` in `src/eosquality/library/identity.py`). Its molecules are the [canonical Ersilia reference library](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs).

The same name is used everywhere:
- the `library_name` field in the index's `metadata.json`
- the source CSV stem under `data/libraries/`
- the index folder under `data/indices/`
- the user-cache folder
- the S3 path segment

The library's `vN` must equal the package's major version. Importing a release where they disagree raises an error.

**When to bump.** Any change that alters scores needs a new library, and therefore a new package major version. That covers adding or removing molecules, correcting SMILES, and changing the Morgan parameters. Edits to metadata only do not.

**Status.** The package is currently `0.0.1` with library `ersilia_reference_library_v0` (about 1.36M molecules).

## Artifact compatibility

Saved artifacts record the following in `reference_mode/shared/metadata.json`:
- `library_id`
- `eosquality_version`
- `format_version` (the on-disk layout and score semantics; currently 6)
- `vector_index_path` (for custom indices)

The training modality is versioned separately: `training_mode/training_sets/metadata.json` holds `training_format_version` (currently 1). Adding or changing the training modality therefore never invalidates reference artifacts.

`ErsiliaQuality.load` rejects artifacts in these cases:
- the format version is different, or the folder uses the old flat layout (no `reference_mode/` / `training_mode/`) → `ArtifactVersionError`; refit;
- the library is not this install's canonical library and no custom index path is recorded → `IncompatibleArtifactsError`;
- the package major version is different → `IncompatibleArtifactsError`.

When an artifact is fit on the canonical library, only its identity is stored, not a path, so the artifact is portable between machines. The library is resolved again at run time (see [cli.md](cli.md#the-reference-library)).

When an artifact is fit against a custom index (`vector_index=` in the Python API), the absolute path of that index is stored, and the folder must still exist when `run` is called.

## Library folder contents

| file | content |
|---|---|
| `vector_index.h5` | FPSim2 Morgan database (radius 2, 2048 bits) |
| `knn_indices.npy`, `knn_distances.npy` | `(n, 50)` self-kNN, each molecule's own row excluded |
| `smiles.csv` | the SMILES in index order |
| `metadata.json` | library name, Morgan parameters, `max_k`, SMILES digest, RDKit/FPSim2 versions |
| `physchem_scaled.npy`, `physchem_scaler.json` | RDKit physchem descriptors, median-imputed and standard-scaled (float16), plus the scaler parameters |
| `maccs.npy` | `(n, 166)` MACCS keys |

**Version pinning.** Results depend on the RDKit version, because both Morgan bits and the list of physchem descriptors change between releases:
- loading the index checks the RDKit version recorded in `metadata.json`;
- the Signal physchem backend checks the descriptor list recorded in `physchem_scaler.json`.

## Releasing a new library (maintainers)

1. Bump `LIBRARY_ID` and the package major version together.
2. Build the index from the new Ersilia SMILES CSV:
   ```bash
   eosquality build --input data/libraries/ersilia_reference_library_v1.csv \
                    --output data/indices/ersilia_reference_library_v1/
   ```
   Self-kNN is the slow step, roughly 1M top-k queries. The build can be resumed.
3. Upload the CSV and the folder with [eosvc](https://github.com/ersilia-os/eosvc). The repo's `access.json` routes `data/` to the public bucket:
   ```bash
   eosvc upload --path data/libraries/ersilia_reference_library_v1.csv
   eosvc upload --path data/indices/ersilia_reference_library_v1/
   ```
