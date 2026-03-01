from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from .schema_text import prompt_from_instruction
except ImportError:
    from data.schema_text import prompt_from_instruction


def _safe_import_h5py():
    try:
        import h5py

        return h5py
    except Exception:
        return None


h5py = _safe_import_h5py()


def parse_obs_ptr(obs_ptr: str) -> Tuple[str, str, Optional[int]]:
    if "#" not in obs_ptr:
        return obs_ptr, "", None
    file_path, inner = obs_ptr.split("#", 1)
    parts = [p for p in inner.split("/") if p]
    if not parts:
        return file_path, "", None
    try:
        frame_idx = int(parts[-1])
        dataset_path = "/" + "/".join(parts[:-1])
        return file_path, dataset_path, frame_idx
    except ValueError:
        return file_path, "/" + "/".join(parts), None


def split_dataset_path(dataset_path: str) -> Tuple[str, str]:
    parts = [p for p in dataset_path.split("/") if p]
    if len(parts) < 2:
        return dataset_path, ""
    return "/" + "/".join(parts[:-1]), parts[-1]


def resize_image_nn(img: np.ndarray, out_hw: int) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        return np.zeros((out_hw, out_hw, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_hw, out_hw, 3), dtype=np.uint8)
    ys = np.linspace(0, h - 1, out_hw).astype(np.int32)
    xs = np.linspace(0, w - 1, out_hw).astype(np.int32)
    out = img[ys][:, xs]
    if out.dtype != np.uint8:
        if out.max() <= 1.0:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape[-1] > 3:
        out = out[..., :3]
    return out


class SchemaSFTDataset:
    def __init__(
        self,
        index_path: str,
        schema_text_path: str,
        instr_wrong_path: Optional[str] = None,
        image_size: int = 224,
        split_path: Optional[str] = None,
        split_name: Optional[str] = None,
    ):
        self._h5_cache: Dict[str, object] = {}
        self._dataset_len_cache: Dict[Tuple[str, str], int] = {}
        self.image_size = int(image_size)

        index_df = pd.read_parquet(index_path).copy()
        index_df["sample_id"] = index_df["episode_id"].astype(str) + ":" + index_df["frame_id"].astype(int).astype(str)
        schema_df = pd.read_parquet(schema_text_path).copy()
        self.df = index_df.merge(
            schema_df,
            on=["sample_id", "dataset", "episode_id", "frame_id"],
            how="inner",
        )
        if instr_wrong_path:
            instr_df = pd.read_parquet(instr_wrong_path).copy()
            if "sample_id" not in instr_df.columns and {"episode_id", "frame_id"}.issubset(instr_df.columns):
                instr_df["sample_id"] = instr_df["episode_id"].astype(str) + ":" + instr_df["frame_id"].astype(int).astype(str)
            keep_cols = [c for c in ["sample_id", "instr_blank", "instr_shuffle", "instr_swap", "swap_valid", "swap_meta"] if c in instr_df.columns]
            self.df = self.df.merge(instr_df[keep_cols], on="sample_id", how="left")
        if split_path and split_name:
            allowed = self._load_split_episodes(split_path, split_name)
            self.df = self.df[self.df["episode_id"].astype(str).isin(allowed)].reset_index(drop=True)

    def _load_split_episodes(self, split_path: str, split_name: str) -> Set[str]:
        payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
        split_map = payload.get("splits", {})
        episodes = split_map.get(split_name, [])
        if not isinstance(episodes, list):
            return set()
        return {str(x) for x in episodes}

    def __len__(self) -> int:
        return len(self.df)

    def _get_h5(self, file_path: str):
        if h5py is None:
            return None
        key = str(Path(file_path))
        if key not in self._h5_cache:
            try:
                self._h5_cache[key] = h5py.File(file_path, "r")
            except Exception:
                return None
        return self._h5_cache.get(key)

    def _read_image(self, row: pd.Series) -> np.ndarray:
        obs_ptr = str(row.get("obs_ptr", ""))
        file_path, dataset_path, frame_idx = parse_obs_ptr(obs_ptr)
        if not file_path or not dataset_path or frame_idx is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        h5f = self._get_h5(file_path)
        if h5f is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            ds = h5f[dataset_path]
            img = np.array(ds[frame_idx])
        except Exception:
            dataset_base, fallback_view = split_dataset_path(dataset_path)
            if dataset_base and fallback_view:
                try:
                    img = np.array(h5f[f"{dataset_base}/{fallback_view}"][frame_idx])
                except Exception:
                    return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return resize_image_nn(img, self.image_size)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        instruction = str(row.get("instruction", ""))
        prompt = prompt_from_instruction(instruction)
        target = str(row.get("schema_text", ""))
        return {
            "sample_id": str(row.get("sample_id", "")),
            "dataset": str(row.get("dataset", "")),
            "episode_id": str(row.get("episode_id", "")),
            "frame_id": int(row.get("frame_id", 0)),
            "instruction": instruction,
            "prompt_text": prompt,
            "target_schema_text": target,
            "image": self._read_image(row),
            "instr_blank": str(row.get("instr_blank", "")),
            "instr_shuffle": str(row.get("instr_shuffle", "")),
            "instr_swap": str(row.get("instr_swap", "")),
            "swap_valid": bool(row.get("swap_valid", False)),
            "swap_meta": str(row.get("swap_meta", "")),
        }

    def close(self) -> None:
        for handle in self._h5_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_cache = {}

    def __del__(self) -> None:
        self.close()
