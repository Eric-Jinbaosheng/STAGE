import math
from typing import List


EPS = 1e-8


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = safe_norm(a) * safe_norm(b) + EPS
    return dot(a, b) / denom


def sigmoid(x: float) -> float:
    # Mild slope for stable relation scores from noisy inputs.
    return 1.0 / (1.0 + math.exp(-x))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

