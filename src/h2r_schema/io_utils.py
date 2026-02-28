import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def try_write_parquet(rows: List[Dict], parquet_path: Path) -> Tuple[bool, str]:
    if not rows:
        return False, "empty_rows"
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False, "pandas_missing"
    try:
        df = pd.DataFrame(rows)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        return True, "ok"
    except Exception as exc:
        return False, f"parquet_write_failed:{exc}"

