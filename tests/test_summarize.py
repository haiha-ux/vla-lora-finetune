import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from summarize import wilson, boot_diff, row


def test_wilson_bounds_and_known_value():
    lo, hi = wilson(66, 100)
    assert abs(lo - 0.5628) < 1e-3 and abs(hi - 0.7454) < 2e-3     # matches libero-vla-repro's 56.3-74.5
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(0, 10)[0] > -1e-12 and wilson(10, 10)[1] < 1 + 1e-12


def test_row_per_seed_and_pooled():
    r = row({1: [1, 0, 0, 0], 2: [1, 1, 0, 0]})
    assert r["successes"] == 3 and r["episodes"] == 8 and r["per_seed_rate"] == {"1": 0.25, "2": 0.5}
    assert abs(r["mean_of_seeds"] - 0.375) < 1e-9


def test_boot_diff_brackets_true_difference():
    lo, hi = boot_diff([0] * 50 + [1] * 50, [0] * 20 + [1] * 80)
    assert lo < 0.3 < hi and lo > 0
