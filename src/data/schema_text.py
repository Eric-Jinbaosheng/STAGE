from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PRIMITIVE_ORDER = [
    "HOLD",
    "BACKOFF_SMALL",
    "VIEWPOINT_CHANGE",
    "REALIGN",
    "CLOSE_GENTLE",
    "RETRACT",
    "PROMPT",
]


TRI_VALUE_FIELDS = [
    ("SECURED", "object_secured"),
    ("CONTACT", "contact_confirmed"),
    ("OCCLUDED", "occluded"),
]


SCHEMA_TEMPLATE_ORDER = [
    "TARGET",
    "PHASE",
    "SECURED",
    "CONTACT",
    "OCCLUDED",
    "AFF",
]


TARGET_FALLBACK_KEYWORDS = [
    "drawer",
    "stove",
    "bowl",
    "bottle",
    "box",
    "can",
    "plate",
    "pot",
    "mug",
    "basket",
]


def parse_affordance_mask(raw: str) -> Dict[str, int]:
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {}
    out: Dict[str, int] = {}
    for key in PRIMITIVE_ORDER:
        out[key] = 1 if int(payload.get(key, 0)) else 0
    return out


def affordance_mask_to_text(mask: Dict[str, int]) -> str:
    parts = [f"{key}={1 if int(mask.get(key, 0)) else 0}" for key in PRIMITIVE_ORDER]
    return "[" + ",".join(parts) + "]"


def normalize_tri_value(value: str) -> str:
    v = str(value or "UNK").strip().upper()
    if v in ("T", "F", "UNK"):
        return v
    if v in ("TRUE", "1"):
        return "T"
    if v in ("FALSE", "0"):
        return "F"
    return "UNK"


def row_to_schema_fields(row: Dict) -> Dict[str, str]:
    mask = parse_affordance_mask(str(row.get("affordance_mask", "{}")))
    fields = {
        "TARGET": str(row.get("target_object", "UNK") or "UNK"),
        "PHASE": str(row.get("phase", "UNK") or "UNK"),
        "AFF": affordance_mask_to_text(mask),
    }
    for out_key, in_key in TRI_VALUE_FIELDS:
        fields[out_key] = normalize_tri_value(str(row.get(in_key, "UNK")))
    return fields


def schema_fields_to_text(fields: Dict[str, str]) -> str:
    lines = []
    for key in SCHEMA_TEMPLATE_ORDER:
        value = str(fields.get(key, "UNK"))
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def row_to_schema_text(row: Dict) -> str:
    return schema_fields_to_text(row_to_schema_fields(row))


def parse_schema_text(text: str) -> Dict[str, object]:
    full_text = str(text)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    parsed: Dict[str, object] = {}
    raw_map: Dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw_map[key.strip().upper()] = value.strip()

    parsed["TARGET"] = raw_map.get("TARGET", "UNK")
    parsed["PHASE"] = raw_map.get("PHASE", "UNK")
    for out_key, _ in TRI_VALUE_FIELDS:
        parsed[out_key] = normalize_tri_value(raw_map.get(out_key, "UNK"))

    aff_raw = raw_map.get("AFF", "")
    aff_map: Dict[str, int] = {k: 0 for k in PRIMITIVE_ORDER}
    match = re.search(r"\[(.*)\]", aff_raw)
    inner = match.group(1) if match else aff_raw
    for chunk in [x.strip() for x in inner.split(",") if x.strip()]:
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().upper()
        if key in aff_map:
            aff_map[key] = 1 if value.strip() in ("1", "T", "TRUE", "true") else 0
    parsed["AFF"] = aff_map

    # Fallback parsing for common non-template generations observed from the VLM.
    lowered = full_text.lower()
    if str(parsed["TARGET"]).upper() == "UNK":
        object_match = re.search(r"['\"]object['\"]\s*:\s*['\"]([^'\"]+)['\"]", full_text, re.IGNORECASE)
        object_text = object_match.group(1).lower() if object_match else lowered
        for keyword in TARGET_FALLBACK_KEYWORDS:
            if keyword in object_text:
                parsed["TARGET"] = keyword
                break
        if str(parsed["TARGET"]).upper() == "UNK":
            for keyword in TARGET_FALLBACK_KEYWORDS:
                if keyword in lowered:
                    parsed["TARGET"] = keyword
                    break

    if str(parsed["PHASE"]).upper() == "UNK":
        if "open" in lowered or "turn on" in lowered or "_open_" in lowered or lowered.startswith("open_"):
            parsed["PHASE"] = "manipulate"
        elif "grasp" in lowered or "pick up" in lowered:
            parsed["PHASE"] = "grasp"
        elif "place" in lowered or "put " in lowered:
            parsed["PHASE"] = "place"
        elif "reach" in lowered:
            parsed["PHASE"] = "reach"

    return parsed


def schema_text_to_label_dict(text: str) -> Dict[str, object]:
    parsed = parse_schema_text(text)
    return {
        "target_object": str(parsed.get("TARGET", "UNK")),
        "phase": str(parsed.get("PHASE", "UNK")),
        "object_secured": str(parsed.get("SECURED", "UNK")),
        "contact_confirmed": str(parsed.get("CONTACT", "UNK")),
        "occluded": str(parsed.get("OCCLUDED", "UNK")),
        "affordance_mask": parsed.get("AFF", {k: 0 for k in PRIMITIVE_ORDER}),
    }


def dataframe_to_schema_text(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        sample_id = row_dict.get("sample_id")
        if sample_id is None and {"episode_id", "frame_id"}.issubset(row.index):
            sample_id = f"{row_dict['episode_id']}:{int(row_dict['frame_id'])}"
        rows.append(
            {
                "sample_id": str(sample_id or ""),
                "dataset": str(row_dict.get("dataset", "")),
                "episode_id": str(row_dict.get("episode_id", "")),
                "frame_id": int(row_dict.get("frame_id", 0)),
                "schema_text": row_to_schema_text(row_dict),
            }
        )
    return pd.DataFrame(rows)


def prompt_from_instruction(instruction: str) -> str:
    return (
        "You are a robot policy assistant.\n"
        "Given the instruction and the current observation, output the interaction schema in the exact format.\n"
        "Output exactly six lines.\n"
        "Do not output JSON.\n"
        "Do not output code.\n"
        "Do not explain anything.\n"
        "Use this exact field order:\n"
        "TARGET=<...>\n"
        "PHASE=<...>\n"
        "SECURED=<T|F|UNK>\n"
        "CONTACT=<T|F|UNK>\n"
        "OCCLUDED=<T|F|UNK>\n"
        "AFF=[HOLD=0/1,BACKOFF_SMALL=0/1,VIEWPOINT_CHANGE=0/1,REALIGN=0/1,CLOSE_GENTLE=0/1,RETRACT=0/1,PROMPT=0/1]\n\n"
        f"INSTRUCTION: {instruction}\n"
        "OUTPUT_SCHEMA:\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serialize schema labels into a fixed text template.")
    p.add_argument("--labels", default="data/processed/schema_labels.parquet")
    p.add_argument("--out", default="data/processed/schema_text.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_parquet(args.labels)
    if "sample_id" not in labels.columns and {"episode_id", "frame_id"}.issubset(labels.columns):
        labels = labels.copy()
        labels["sample_id"] = labels["episode_id"].astype(str) + ":" + labels["frame_id"].astype(int).astype(str)
    out_df = dataframe_to_schema_text(labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
