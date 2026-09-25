"""The fitter must recover a temperature it did not know about.

Synthetic, no model download: build logits, apply a known temperature to get the
"shipped" probabilities, then check that fitting recovers the distortion and that the
argmax never moves.
"""
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.calibrate import fit_scale, fit_temperatures, nll  # noqa: E402


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _synth(n=400, k=4, true_scale=3.0, seed=0):
    """Probabilities that are under-confident by `true_scale`, with honest labels."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        z = rng.normal(0.0, 1.5, size=k)
        honest = _softmax(z)
        gold = int(rng.choice(k, p=honest))
        # what a miscalibrated checkpoint would have reported: flattened by true_scale
        shipped = _softmax(z / true_scale)
        records.append((shipped, gold))
    return records


def test_recovers_known_scale():
    records = _synth(n=2000, true_scale=3.0)
    s = fit_scale(records)
    assert abs(s - 3.0) < 0.35, "expected to recover ~3.0, got %.3f" % s


def test_fitting_reduces_nll():
    records = _synth(n=1000, true_scale=2.5)
    s = fit_scale(records)
    assert nll(records, s) < nll(records, 1.0)


def test_argmax_is_invariant():
    """Temperature scaling must never change the decision, only its probability."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(2, 9))
        p = _softmax(rng.normal(0.0, 1.5, size=k))
        for s in (0.2, 0.5, 1.0, 2.0, 7.0):
            q = np.power(p, s)
            q /= q.sum()
            assert int(q.argmax()) == int(p.argmax())
            assert abs(q.sum() - 1.0) < 1e-9


def test_degenerate_inputs_do_not_explode():
    assert math.isfinite(nll([(np.array([1.0, 0.0]), 0)], 1.0))
    assert math.isfinite(nll([(np.array([1.0, 0.0]), 1)], 1.0))
    assert nll([], 1.0) == 0.0


class _StubAgent:
    """Replays cached noul probabilities, as `laya-typed-decisions` returns them at T ~1.0."""

    def __init__(self, records, lang_temperatures=None):
        self.records = records
        self.temperature = [1.0, 1.0, 1.0]
        self.temperature_by_options = {}
        self.lang_temperatures = lang_temperatures or {}
        self.langs = []

    def predict(self, state, questions, lang=None):
        self.langs.append(lang)
        p, _ = self.records[state]
        return {"answers": {"q": {"type": "noul", "probabilities": {"false": p[0], "true": p[1]}}}}


def _examples(records):
    return [(i, {"q": {"type": "noul"}}, {"q": ["false", "true"][g]}) for i, (_, g) in enumerate(records)]


def test_fit_stays_inside_the_runtime_clamp():
    """Under-confident by 4x wants T = 0.25; the runtime refuses anything below 0.5."""
    records = _synth(n=1000, k=2, true_scale=4.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = fit_temperatures(_StubAgent(records), _examples(records))
    assert abs(cfg["temperature_by_options"]["noul:2"] - 0.5) < 1e-3, cfg
    assert any("noul:2" in str(w.message) for w in caught), "a pinned bucket must be reported"


def test_inside_the_clamp_no_warning():
    records = _synth(n=1000, k=2, true_scale=1.6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = fit_temperatures(_StubAgent(records), _examples(records))
    assert 0.5 < cfg["temperature_by_options"]["noul:2"] < 0.8, cfg
    assert not caught, [str(w.message) for w in caught]


def test_lang_fits_from_that_languages_temperatures():
    """With `lang`, t0 is the language override and predictions are made under it."""
    records = _synth(n=1000, k=2, true_scale=1.0)       # honest probabilities: keep t0
    agent = _StubAgent(records, {"de": {"temperature": [2.0, 2.0, 2.0], "temperature_by_options": {}}})
    cfg = fit_temperatures(agent, _examples(records), lang="de-AT")
    assert abs(cfg["temperature_by_options"]["noul:2"] - 2.0) < 0.3, cfg
    assert set(agent.langs) == {"de-AT"}


if __name__ == "__main__":
    test_recovers_known_scale()
    test_fitting_reduces_nll()
    test_argmax_is_invariant()
    test_degenerate_inputs_do_not_explode()
    test_fit_stays_inside_the_runtime_clamp()
    test_inside_the_clamp_no_warning()
    test_lang_fits_from_that_languages_temperatures()
    print("all calibration checks passed")
