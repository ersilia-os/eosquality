import numpy as np
import pytest

from eosquality.library.physchem import (
    DESCRIPTOR_FNS,
    DESCRIPTOR_NAMES,
    N_DESCRIPTORS,
    _compute_one,
    check_descriptor_names,
    compute_physchem_raw,
)


def test_descriptors_equal_rdkit_s_own(smiles):
    """The shared Ipc/AvgIpc polynomial must not change a single value."""
    from rdkit import Chem

    for smi in smiles[:25]:
        mol = Chem.MolFromSmiles(smi)
        expected = np.array([float(fn(mol)) for _, fn in DESCRIPTOR_FNS])
        got = _compute_one(smi).astype(np.float64)
        finite = np.isfinite(expected) & (np.abs(expected) < 3e38)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-5, atol=1e-6)
        assert np.isnan(got[~finite]).all()


def test_an_unparsable_smiles_gives_a_row_of_nan():
    row = _compute_one("not a smiles")
    assert row.shape == (N_DESCRIPTORS,) and np.isnan(row).all()


def test_the_matrix_has_one_row_per_molecule_in_order():
    raw = compute_physchem_raw(["CCO", "c1ccccc1", "CCO"])
    assert raw.shape == (3, N_DESCRIPTORS)
    np.testing.assert_array_equal(raw[0], raw[2])
    assert not np.array_equal(raw[0], raw[1])
    assert compute_physchem_raw([]).shape == (0, N_DESCRIPTORS)


def test_a_descriptor_list_that_changed_is_refused():
    check_descriptor_names({"descriptor_names": list(DESCRIPTOR_NAMES)})
    with pytest.raises(RuntimeError, match="descriptor list mismatch"):
        check_descriptor_names({"descriptor_names": DESCRIPTOR_NAMES[:-1]})
