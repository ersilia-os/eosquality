"""The CLI and ``import eosquality`` must not import the scientific stack."""

import subprocess
import sys

HEAVY = ("pandas", "sklearn", "scipy", "rdkit", "xgboost", "FPSim2", "eosframes")


def _loaded_heavy(code: str) -> list[str]:
    probe = (
        f"{code}\nimport sys\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return [m for m in out.stdout.strip().split(",") if m]


def test_cli_start_up_is_light():
    assert _loaded_heavy("import eosquality.cli") == []


def test_package_import_is_light_until_a_class_is_used():
    assert _loaded_heavy("import eosquality") == []
    assert "pandas" in _loaded_heavy("from eosquality import ErsiliaQuality")


def test_lazy_names_resolve():
    import eosquality

    assert eosquality.ErsiliaQuality.__name__ == "ErsiliaQuality"
    assert eosquality.RunResult.__name__ == "RunResult"
    assert set(eosquality.__all__) >= {"ErsiliaQuality", "Support", "set_verbosity"}
