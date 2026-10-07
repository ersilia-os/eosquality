"""Shared fixtures: a tiny custom library (SMILES + match keys) and synthetic model outputs.

Built once per test session from ``tests/fixtures/smiles_700.csv`` (a random
sample of the canonical reference library): the first 600 molecules form
the reference, the last 100 are novel queries. Model outputs are simple
RDKit descriptors plus noise, with a few NaNs, so every score has
something to calibrate.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from eosquality.cli.build import build_library

DATA = pathlib.Path(__file__).parent / "fixtures"
N_REF = 600


def model_outputs(smiles: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                rdMolDescriptors.CalcTPSA(mol),
                float(rdMolDescriptors.CalcNumAromaticRings(mol) > 0),
                rdMolDescriptors.CalcNumHBD(mol),
            ]
        )
    df = pd.DataFrame(rows, columns=["mw", "logp", "tpsa", "aromatic", "hbd"])
    df["noisy"] = df["logp"] + rng.normal(0, 1, len(df))
    df.loc[rng.choice(len(df), size=len(df) // 50, replace=False), "tpsa"] = np.nan
    df.insert(0, "input", smiles)
    df.insert(0, "key", [f"k{seed}_{i}" for i in range(len(df))])
    return df


@pytest.fixture(scope="session")
def make_outputs():
    return model_outputs


@pytest.fixture(scope="session")
def smiles() -> list[str]:
    return list(pd.read_csv(DATA / "smiles_700.csv")["smiles"])


@pytest.fixture(scope="session")
def library(tmp_path_factory, smiles) -> pathlib.Path:
    """Custom library folder (SMILES, metadata, match keys) of the first N_REF molecules."""
    out = tmp_path_factory.mktemp("library")
    build_library(smiles[:N_REF], out, "test_library")
    return out


@pytest.fixture(scope="session")
def reference(smiles) -> pd.DataFrame:
    return model_outputs(smiles[:N_REF], seed=0)


@pytest.fixture(scope="session")
def query(smiles) -> pd.DataFrame:
    """60 novel molecules + 40 molecules that are in the reference."""
    return model_outputs(smiles[N_REF : N_REF + 60] + smiles[:40], seed=1)


@pytest.fixture(scope="session")
def training_dir(tmp_path_factory, smiles) -> pathlib.Path:
    """Training sets for three output columns of the fixture model.

    ``mw``, ``aromatic`` and ``hbd``, drawn from the 700-molecule fixture so
    some overlap the reference and queries. ``mw`` also carries a salt form
    and an exact duplicate of a molecule to exercise standardisation and
    merging.
    """
    folder = tmp_path_factory.mktemp("training") / "training_eos0aaa_v1"
    folder.mkdir()
    outputs = model_outputs(smiles, seed=2)
    mw = outputs.iloc[100:400][["input"]].rename(columns={"input": "smiles"})
    extra = pd.DataFrame({"smiles": [mw.smiles.iloc[0] + ".Cl", mw.smiles.iloc[1]]})
    pd.concat([mw, extra]).to_csv(folder / "mw.csv", index=False)
    for name, rows in (("aromatic", slice(300, 650)), ("hbd", slice(0, 250))):
        outputs.iloc[rows][["input"]].rename(columns={"input": "smiles"}).to_csv(
            folder / f"{name}.csv", index=False
        )
    return folder
