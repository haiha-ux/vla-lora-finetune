import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from svla import to_libero_gripper, DS_TO_LIBERO, LIBERO_TO_DS, normalize


def test_gripper_convention_conversion():
    a = np.zeros((2, 3, 7), np.float32); a[..., 6] = [[1, 0, 1], [0, 0, 1]]
    out = to_libero_gripper(a)
    assert out[..., 6].tolist() == [[-1, 1, -1], [1, 1, -1]]   # open(1) -> -1, close(0) -> +1
    assert (out[..., :6] == 0).all() and a[0, 0, 6] == 1        # input not modified


def test_task_mapping_is_a_bijection():
    assert sorted(DS_TO_LIBERO.values()) == list(range(10))
    assert all(LIBERO_TO_DS[DS_TO_LIBERO[k]] == k for k in range(10))
    assert LIBERO_TO_DS[5] == 2   # "on the ramekin"


def test_normalize_round_trip():
    x = np.array([1.0, 2.0]); m, s = 0.5, 2.0
    assert np.allclose(normalize(x, m, s) * (s + 1e-8) + m, x)
