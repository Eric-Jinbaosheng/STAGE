import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_OBJECT_SYNONYMS: Dict[str, List[str]] = {
    "mug": ["mug", "cup"],
    "bowl": ["bowl", "ramekin"],
    "can": ["soda can", "alphabet soup", "can"],
    "bottle": ["wine bottle", "tomato sauce", "bbq sauce", "salad dressing", "orange juice", "ketchup", "milk", "bottle"],
    "box": ["cookie box", "chocolate pudding", "cream cheese", "butter", "box"],
    "plate": ["plate", "dish"],
    "pot": ["moka pots", "moka pot", "frying pan", "kettle", "pots", "pot", "pan"],
    "basket": ["basket", "caddy"],
    "drawer": ["drawer", "cabinet"],
    "stove": ["burner", "stove"],
    "book": ["book"],
    "spoon": ["spoon"],
    "block": ["block"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_object_synonyms(path: str = "") -> Dict[str, List[str]]:
    if path:
        vocab_path = Path(path)
        if vocab_path.exists():
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if "object_vocab" in payload and isinstance(payload["object_vocab"], dict):
                    return {str(k): [str(x) for x in v] for k, v in payload["object_vocab"].items()}
                if "object_vocab" in payload and isinstance(payload["object_vocab"], list):
                    return {str(x): [str(x)] for x in payload["object_vocab"]}
                return {str(k): [str(x) for x in v] for k, v in payload.items()}
    return dict(DEFAULT_OBJECT_SYNONYMS)


def build_alias_table(object_synonyms: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    aliases: List[Tuple[str, str]] = []
    for canonical, syns in object_synonyms.items():
        seen = set()
        for item in [canonical, *syns]:
            norm = _normalize(item)
            if norm and norm not in seen:
                aliases.append((norm, canonical))
                seen.add(norm)
    aliases.sort(key=lambda x: (-len(x[0]), x[0]))
    return aliases


def extract_object_mentions(text: str, object_synonyms: Dict[str, List[str]]) -> List[str]:
    low = _normalize(text)
    aliases = build_alias_table(object_synonyms)
    found: List[str] = []
    occupied = [False] * len(low)
    for alias, canonical in aliases:
        for match in re.finditer(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", low):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            found.append(canonical)
            for i in range(start, end):
                occupied[i] = True
            break
    # Preserve first appearance order.
    deduped: List[str] = []
    for item in found:
        if item not in deduped:
            deduped.append(item)
    return deduped


def parse_target_object(text: str, object_synonyms: Dict[str, List[str]]) -> str:
    mentions = extract_object_mentions(text, object_synonyms)
    return mentions[0] if mentions else "UNK"


def swap_target_in_instruction(text: str, src: str, tgt: str, object_synonyms: Dict[str, List[str]]) -> str:
    if not text or src == "UNK" or not tgt:
        return text
    low = _normalize(text)
    aliases = build_alias_table(object_synonyms)
    src_aliases = [alias for alias, canonical in aliases if canonical == src]
    if not src_aliases:
        return text
    src_aliases.sort(key=len, reverse=True)
    for alias in src_aliases:
        match = re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", low)
        if not match:
            continue
        start, end = match.span()
        return text[:start] + tgt + text[end:]
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse instruction into target object and mentions.")
    p.add_argument("--instruction", required=True)
    p.add_argument("--object-vocab", default="", help="Optional object vocab JSON with synonyms.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    object_synonyms = load_object_synonyms(args.object_vocab)
    mentions = extract_object_mentions(args.instruction, object_synonyms)
    payload = {
        "instruction": args.instruction,
        "target_object": parse_target_object(args.instruction, object_synonyms),
        "mentions": mentions,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
