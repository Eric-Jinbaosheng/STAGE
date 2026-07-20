from __future__ import annotations

from typing import Any, Dict, Iterable, List


class BaseSchemaModel:
    name = "base"
    available = True
    unavailable_reason = ""

    def predict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_many(self, examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(x) for x in examples]


class UnavailableModel(BaseSchemaModel):
    available = False

    def __init__(self, name: str, reason: str):
        self.name = name
        self.unavailable_reason = reason

    def predict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(f"{self.name} unavailable: {self.unavailable_reason}")
