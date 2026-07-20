from __future__ import annotations

import random
from typing import Iterable, Tuple


def bootstrap_mean_ci(values: Iterable[float], seed: int = 7, rounds: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((alpha / 2.0) * (len(means) - 1))]
    hi = means[int((1.0 - alpha / 2.0) * (len(means) - 1))]
    return lo, hi
