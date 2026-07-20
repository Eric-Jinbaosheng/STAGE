import re
from typing import Iterable, List, Optional

SAFE_ACTIONS = {"ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF", "TARGET_NOT_FOUND"}
TASK_ACTIONS = {"GRASP", "PICK", "PLACE", "RETRACT", "MOVE_TO", "OPEN", "CLOSE", "TRANSFER"}
ALL_ACTIONS = sorted(SAFE_ACTIONS | TASK_ACTIONS)

_OBJECT_WORDS = [
    "bottle", "bowl", "box", "can", "drawer", "plate", "stove", "object_0", "mug", "cup", "cube", "book"
]

_PROHIBITION_PATTERNS = [
    r"\bdo\s+not\b", r"\bdon't\b", r"\bavoid\b", r"\bnever\b", r"\bmust\s+not\b", r"\bno\s+\w+ing\b",
    r"\bwait\s+before\b", r"\buntil\s+released\b",
]


def normalize_action(action: object) -> str:
    text = str(action or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "ASK_HUMAN": "ASK",
        "PROMPT_HUMAN": "PROMPT",
        "PROMPT_HUMAN_RELEASE": "PROMPT",
        "HOLD_COMPLIANT": "HOLD",
        "BACKOFF_SMALL": "BACKOFF",
        "CLOSE_GENTLE": "GRASP",
        "TARGET_NOT_FOUND": "TARGET_NOT_FOUND",
        "NONE": "WAIT",
        "": "WAIT",
    }
    return aliases.get(text, text)


def is_blank_instruction(text: object) -> bool:
    return len(str(text or "").strip()) == 0


def has_prohibition(text: object) -> bool:
    low = str(text or "").lower()
    return any(re.search(p, low) for p in _PROHIBITION_PATTERNS)


def extract_target(text: object, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    low = str(text or "").lower().replace("-", "_")
    search = list(candidates or _OBJECT_WORDS)
    search = sorted({str(x).lower() for x in search if str(x)}, key=len, reverse=True)
    for obj in search:
        if obj and re.search(rf"(?<![a-z0-9_]){re.escape(obj)}(?![a-z0-9_])", low):
            return obj
    return None


def choose_impossible_target(scene_objects: Iterable[str]) -> str:
    scene = {str(x).lower() for x in scene_objects}
    for cand in ["red_mug", "purple_cube", "green_book", "silver_spoon"]:
        if cand not in scene:
            return cand
    return "nonexistent_target"


def action_is_task_specific(action: object) -> bool:
    return normalize_action(action) in TASK_ACTIONS


def safe_deferral_action(action: object) -> bool:
    return normalize_action(action) in SAFE_ACTIONS


def action_mentions_target(action: object, target: object) -> bool:
    a = str(action or "").lower()
    t = str(target or "").lower()
    return bool(t and t in a)
