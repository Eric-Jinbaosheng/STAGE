import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List


DEFAULT_ITEMS = [
    "README.md",
    ".gitignore",
    "schema/schema_min.json",
    "scripts/package_transfer_bundle.py",
    "scripts/train_finetune_baseline.py",
    "src/data/__init__.py",
    "src/data/common_io.py",
    "src/data/build_index_libero.py",
    "src/data/build_index_handover.py",
    "src/data/build_schema_labels.py",
    "src/data/dataloader.py",
    "src/data/viz_sanity.py",
    "docs/10_training_machine_migration.md",
    "docs/11_github_repo_data_and_finetune_setup.md",
    "docs/12_finetune_baseline_skeleton.md",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package a code-only transfer bundle for the training machine.")
    p.add_argument("--out-dir", default="transfer_bundle", help="Directory to populate with bundled files.")
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional repo-relative file to include. Can be passed multiple times.",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete the output directory before rebuilding the bundle.",
    )
    return p.parse_args()


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_item(repo_root: Path, out_dir: Path, rel_path: str) -> dict:
    src = repo_root / rel_path
    if not src.exists():
        return {"path": rel_path, "status": "missing"}
    dst = out_dir / rel_path
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return {"path": rel_path, "status": "copied", "size_bytes": src.stat().st_size}


def write_manifest(out_dir: Path, manifest: dict) -> None:
    ensure_parent(out_dir / "MANIFEST.json")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    out_dir = (repo_root / args.out_dir).resolve()
    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = dedupe(DEFAULT_ITEMS + args.include)
    results = [copy_item(repo_root, out_dir, rel_path) for rel_path in items]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "bundle_root": str(out_dir),
        "git_commit": get_git_commit(repo_root),
        "items": results,
        "note": "Code-only bundle. Data artifacts are intentionally excluded.",
    }
    write_manifest(out_dir, manifest)

    copied = sum(1 for x in results if x["status"] == "copied")
    missing = [x["path"] for x in results if x["status"] == "missing"]
    print(f"Created transfer bundle: {out_dir}")
    print(f"Copied files: {copied}")
    if missing:
        print("Missing files:")
        for rel_path in missing:
            print(f"  - {rel_path}")


if __name__ == "__main__":
    main()
