import argparse
import json
from pathlib import Path

import pandas as pd

from common_io import write_json
from parse_instruction import DEFAULT_OBJECT_SYNONYMS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a canonical object vocabulary with synonym mapping.")
    p.add_argument("--index", default="data/processed/libero_index.parquet")
    p.add_argument("--out", default="data/processed/object_vocab.json")
    p.add_argument("--meta-out", default="data/processed/meta.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"object_vocab": DEFAULT_OBJECT_SYNONYMS}
    out_path = Path(args.out)
    write_json(out_path, payload)

    meta_path = Path(args.meta_out)
    meta_patch = {
        "object_vocab_path": out_path.as_posix(),
        "object_vocab_size": len(DEFAULT_OBJECT_SYNONYMS),
        "object_vocab_type": "canonical_with_synonyms",
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.update(meta_patch)
        write_json(meta_path, old)
    else:
        write_json(meta_path, meta_patch)

    if Path(args.index).exists():
        df = pd.read_parquet(args.index)
        print(f"Loaded index rows={len(df)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
