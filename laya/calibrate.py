"""Fit per-bucket temperatures on labelled data.

The model divides logits by a temperature before the softmax, so a temperature only
moves the probabilities -- never the argmax. Fitting one is therefore free accuracy-wise
and is the cheapest way to repair calibration on a checkpoint that shipped without it.

Only the returned probabilities are needed, not the logits. If the agent produced `p`
under temperature `t0`, then under a candidate `t` it would have produced
`normalize(p ** (t0 / t))`, so a single scalar search per bucket is enough.

Usage:

    from laya.calibrate import fit_temperatures
    cfg = fit_temperatures(agent, examples)     # examples: (state, questions, golds)
    json.dump(cfg, open("rl_agent_config.json", "w"))

Fit on data the checkpoint was not trained on, and report on data it was not fitted on.

The search stays inside the runtime's `[TEMP_MIN, TEMP_MAX]`, because the agent clamps any
temperature outside it at load: a fitted value the runtime would not apply is not a fit. A
bucket that ends on that bound is reported, since its calibration is then only partly fixed.
"""
import json
import math
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .common import QTYPES, TEMP_MAX, TEMP_MIN, temp_bucket

__all__ = ["fit_temperatures", "fit_scale", "nll"]

_LO, _HI = 0.05, 20.0
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _rescale(p: np.ndarray, s: float) -> np.ndarray:
    q = np.power(np.clip(p, 1e-12, 1.0), s)
    return q / q.sum()


def nll(records: Sequence[Tuple[np.ndarray, int]], s: float) -> float:
    """Mean negative log likelihood of the gold answers after rescaling by `s`."""
    total, n = 0.0, 0
    for p, gold_idx in records:
        if gold_idx < 0:
            continue
        total -= math.log(max(float(_rescale(p, s)[gold_idx]), 1e-12))
        n += 1
    return total / max(n, 1)


def fit_scale(records: Sequence[Tuple[np.ndarray, int]], lo: float = _LO, hi: float = _HI,
              tol: float = 1e-4) -> float:
    """Golden-section search for the scale `s` minimising NLL. Convex in practice."""
    a, b = lo, hi
    c, d = b - _INV_PHI * (b - a), a + _INV_PHI * (b - a)
    fc, fd = nll(records, c), nll(records, d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = nll(records, c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = nll(records, d)
    return (a + b) / 2.0


def _collect(agent, examples: Iterable[Tuple], min_per_bucket: int,
             lang: Optional[str]) -> Dict[str, List]:
    buckets = {}  # type: Dict[str, List[Tuple[np.ndarray, int]]]
    for state, questions, golds in examples:
        answers = agent.predict(state, questions, lang=lang)["answers"]
        for qid, ans in answers.items():
            if qid not in golds:
                continue
            probs = ans.get("probabilities")
            if not probs:
                continue
            keys = list(probs.keys())
            p = np.array([float(probs[k]) for k in keys], dtype=np.float64)
            p = np.clip(p, 1e-12, None)
            p /= p.sum()
            gold = str(golds[qid])
            gold_idx = keys.index(gold) if gold in keys else -1
            qt = QTYPES[ans["type"]]
            buckets.setdefault(temp_bucket(qt, len(keys)), []).append((p, gold_idx))
    return {b: r for b, r in buckets.items() if len(r) >= min_per_bucket}


def fit_temperatures(agent, examples: Iterable[Tuple], min_per_bucket: int = 50,
                     base: Optional[Sequence[float]] = None, lang: Optional[str] = None) -> Dict:
    """Fit temperatures for `agent` on labelled `examples`.

    `examples` yields `(state, questions, golds)` where `golds` maps a question id to its
    correct label -- the same strings the answers use: a criterion key for `choice`, a
    level index for `score`, "true"/"false" for `noul`.

    Returns a dict with `temperature` and `temperature_by_options`, ready to merge into
    `rl_agent_config.json` -- or, with `lang`, to pass as `lang_temperatures[lang]`, which
    takes the same shape. Buckets with fewer than `min_per_bucket` decisions are left alone
    rather than fitted on noise.
    """
    cfg = {}  # type: Dict
    if lang:
        cfg = (getattr(agent, "lang_temperatures", {}) or {}).get(lang.split("-")[0].lower(), {})
    current = list(base if base is not None else
                   cfg.get("temperature", getattr(agent, "temperature", [1.0, 1.0, 1.0])))
    by_options = dict(cfg.get("temperature_by_options", getattr(agent, "temperature_by_options", {})) or {})
    buckets = _collect(agent, examples, min_per_bucket, lang)

    per_qtype = {}  # type: Dict[str, List[float]]
    for bucket, records in sorted(buckets.items()):
        qtype_name = bucket.split(":")[0]
        t0 = float(by_options.get(bucket, current[QTYPES[qtype_name]]))
        # the runtime divides logits by clamp(t), so only s in [t0/TEMP_MAX, t0/TEMP_MIN] is reachable
        lo, hi = t0 / TEMP_MAX, t0 / TEMP_MIN
        s = fit_scale(records, lo, hi)
        t_new = min(TEMP_MAX, max(TEMP_MIN, t0 / s))
        if min(s - lo, hi - s) < 1e-3:
            t_new = TEMP_MIN if hi - s < s - lo else TEMP_MAX
            warnings.warn("laya.calibrate: %s wants a temperature outside [%g, %g] and was fitted "
                          "at the bound %g; its confidence stays partly miscalibrated"
                          % (bucket, TEMP_MIN, TEMP_MAX, t_new))
        by_options[bucket] = round(t_new, 6)
        per_qtype.setdefault(qtype_name, []).append(t_new)

    temperature = list(current)
    for qtype_name, values in per_qtype.items():
        temperature[QTYPES[qtype_name]] = round(float(np.mean(values)), 6)

    return {"temperature": temperature, "temperature_by_options": by_options}


def _main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m laya.calibrate <model_id_or_path> <labelled.jsonl> [out.json]\n\n"
              "Each line: {\"state\": ..., \"questions\": {...}, \"golds\": {\"<q>\": \"<label>\"}}",
              file=sys.stderr)
        return 2
    from . import load

    agent = load(argv[1])
    examples = []
    with open(argv[2]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                examples.append((row["state"], row["questions"], row["golds"]))
    cfg = fit_temperatures(agent, examples)
    out = json.dumps(cfg, indent=2)
    if len(argv) > 3:
        with open(argv[3], "w") as fh:
            fh.write(out + "\n")
        print("wrote %s" % argv[3])
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
