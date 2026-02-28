import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset

        return torch, Dataset, DataLoader
    except Exception:
        return None, object, object


def _safe_import_h5py():
    try:
        import h5py

        return h5py
    except Exception:
        return None


torch, TorchDataset, TorchDataLoader = _safe_import_torch()
h5py = _safe_import_h5py()


def parse_affordance_mask(s: str) -> np.ndarray:
    try:
        d = json.loads(s)
    except Exception:
        d = {}
    keys = ["HOLD", "BACKOFF_SMALL", "VIEWPOINT_CHANGE", "REALIGN", "CLOSE_GENTLE", "RETRACT", "PROMPT"]
    return np.array([float(d.get(k, 0)) for k in keys], dtype=np.float32)


def resize_image_nn(img: np.ndarray, out_hw: int) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        return np.zeros((out_hw, out_hw, 3), dtype=np.float32)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_hw, out_hw, 3), dtype=np.float32)
    if h == out_hw and w == out_hw:
        return img.astype(np.float32)
    ys = np.linspace(0, h - 1, out_hw).astype(np.int32)
    xs = np.linspace(0, w - 1, out_hw).astype(np.int32)
    return img[ys][:, xs].astype(np.float32)


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


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


class UnifiedSchemaDataset(TorchDataset):  # type: ignore[misc]
    def __init__(
        self,
        index_path: str,
        labels_path: str,
        use_image: bool = False,
        image_size: int = 64,
        frame_window: int = 1,
        view_names: Optional[List[str]] = None,
        text_vocab_size: int = 2048,
        text_max_len: int = 16,
    ):
        self.index_df = pd.read_parquet(index_path)
        self.labels_df = pd.read_parquet(labels_path)
        self.df = self.index_df.merge(
            self.labels_df,
            on=["dataset", "episode_id", "frame_id"],
            how="inner",
        )
        self.object_vocab = sorted(self.df["target_object"].astype(str).unique().tolist())
        self.object_to_id = {x: i for i, x in enumerate(self.object_vocab)}
        self.phase_vocab = sorted(self.df["phase"].astype(str).unique().tolist())
        self.phase_to_id = {x: i for i, x in enumerate(self.phase_vocab)}

        self.use_image = bool(use_image)
        self.image_size = int(image_size)
        self.frame_window = max(1, int(frame_window))
        self.text_vocab_size = max(128, int(text_vocab_size))
        self.text_max_len = max(4, int(text_max_len))
        self.view_names = [x for x in (view_names or ["agentview_rgb"]) if x]
        if not self.view_names:
            self.view_names = ["agentview_rgb"]

        self._h5_cache: Dict[str, object] = {}
        self._dataset_len_cache: Dict[Tuple[str, str], int] = {}
        self.text_vocab = self._build_text_vocab()
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self) -> int:
        return len(self.df)

    @property
    def image_channels(self) -> int:
        return 3 * self.frame_window * len(self.view_names)

    def _build_text_vocab(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for text in self.df["instruction"].astype(str).tolist():
            for tok in tokenize_text(text):
                counts[tok] = counts.get(tok, 0) + 1
        # Reserve 0 for pad and 1 for unk.
        items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        keep = items[: max(0, self.text_vocab_size - 2)]
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, (tok, _) in enumerate(keep, start=2):
            vocab[tok] = i
        return vocab

    def encode_instruction(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        toks = tokenize_text(text)
        ids = np.full((self.text_max_len,), self.pad_id, dtype=np.int64)
        mask = np.zeros((self.text_max_len,), dtype=np.float32)
        for i, tok in enumerate(toks[: self.text_max_len]):
            ids[i] = self.text_vocab.get(tok, self.unk_id)
            mask[i] = 1.0
        return ids, mask

    def export_text_vocab(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text_vocab_size": int(self.text_vocab_size),
            "text_max_len": int(self.text_max_len),
            "pad_id": int(self.pad_id),
            "unk_id": int(self.unk_id),
            "vocab": self.text_vocab,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

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

    def _get_dataset_len(self, h5f, dataset_path: str) -> int:
        key = (str(getattr(h5f, "filename", "")), dataset_path)
        if key not in self._dataset_len_cache:
            try:
                self._dataset_len_cache[key] = int(h5f[dataset_path].shape[0])
            except Exception:
                self._dataset_len_cache[key] = 0
        return self._dataset_len_cache[key]

    def _read_single_image(self, h5f, dataset_path: str, frame_idx: int) -> np.ndarray:
        try:
            ds = h5f[dataset_path]
            img = np.array(ds[frame_idx])
        except Exception:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img = resize_image_nn(img, self.image_size)
        if img.max() > 1.0:
            img = img / 255.0
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        return img.astype(np.float32)

    def _load_image_stack(self, row: pd.Series) -> np.ndarray:
        zeros = np.zeros((self.image_channels, self.image_size, self.image_size), dtype=np.float32)
        if not self.use_image:
            return zeros
        if str(row.get("dataset", "")) != "libero":
            return zeros
        obs_ptr = str(row.get("obs_ptr", ""))
        file_path, dataset_path, frame_idx = parse_obs_ptr(obs_ptr)
        if not file_path or not dataset_path or frame_idx is None:
            return zeros
        h5f = self._get_h5(file_path)
        if h5f is None:
            return zeros

        dataset_base, default_view = split_dataset_path(dataset_path)
        dataset_len = self._get_dataset_len(h5f, dataset_path)
        if dataset_len <= 0:
            return zeros

        half = self.frame_window // 2
        offsets = list(range(-half, half + 1))
        if len(offsets) > self.frame_window:
            offsets = offsets[: self.frame_window]
        while len(offsets) < self.frame_window:
            offsets.append(offsets[-1] if offsets else 0)

        planes: List[np.ndarray] = []
        for view in self.view_names:
            view_path = f"{dataset_base}/{view}"
            if view_path not in h5f:
                # Fallback to the current view if requested view is missing.
                view_path = f"{dataset_base}/{default_view}"
            for off in offsets:
                idx = min(max(frame_idx + off, 0), dataset_len - 1)
                img = self._read_single_image(h5f, view_path, idx)
                planes.append(np.transpose(img, (2, 0, 1)))

        if not planes:
            return zeros
        return np.concatenate(planes, axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        feats = np.array(
            [
                float(row.get("d_hand_obj", 0.0)),
                float(row.get("d_grip_obj", 0.0)),
                float(row.get("gripper_width", 0.0)),
                float(row.get("contact_flag", 0.0)),
                float(row.get("visibility", 1.0)),
            ],
            dtype=np.float32,
        )
        aff = parse_affordance_mask(str(row.get("affordance_mask", "{}")))
        img = self._load_image_stack(row)
        input_ids, attention_mask = self.encode_instruction(str(row.get("instruction", "")))
        obj = self.object_to_id[str(row["target_object"])]
        phase = self.phase_to_id[str(row["phase"])]
        out = {
            "x": feats,
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "affordance_mask": aff,
            "label_target_object": obj,
            "label_phase": phase,
            "instruction": str(row.get("instruction", "")),
        }
        if torch is not None:
            out = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in out.items()}
            out["label_target_object"] = torch.tensor(obj, dtype=torch.long)
            out["label_phase"] = torch.tensor(phase, dtype=torch.long)
        return out

    def close(self) -> None:
        for handle in self._h5_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_cache = {}

    def __del__(self) -> None:
        self.close()


def smoke_train(
    index_path: str,
    labels_path: str,
    steps: int = 200,
    batch_size: int = 32,
    use_image: bool = False,
    image_size: int = 64,
    frame_window: int = 1,
    view_names: Optional[List[str]] = None,
    text_vocab_size: int = 2048,
    text_max_len: int = 16,
) -> Tuple[float, float]:
    if torch is None:
        raise RuntimeError("torch is not installed; cannot run smoke train")
    ds = UnifiedSchemaDataset(
        index_path,
        labels_path,
        use_image=use_image,
        image_size=image_size,
        frame_window=frame_window,
        view_names=view_names,
        text_vocab_size=text_vocab_size,
        text_max_len=text_max_len,
    )
    loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    if use_image:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(ds.image_channels, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, max(2, len(ds.object_vocab))),
        )
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(5 + text_max_len, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, max(2, len(ds.object_vocab))),
        )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    first_loss = None
    last_loss = None
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        if use_image:
            x = b["image"].float()
        else:
            x = torch.cat([b["x"].float(), b["attention_mask"].float()], dim=1)
        y = b["label_target_object"].long()
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        last_loss = lv
    ds.close()
    return float(first_loss), float(last_loss)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified dataloader smoke run.")
    p.add_argument("--index", default="data/processed/handover_index.parquet")
    p.add_argument("--labels", default="data/processed/schema_labels.parquet")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-image", action="store_true")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--frame-window", type=int, default=1)
    p.add_argument("--views", default="agentview_rgb")
    p.add_argument("--text-vocab-size", type=int, default=2048)
    p.add_argument("--text-max-len", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    view_names = [x.strip() for x in args.views.split(",") if x.strip()]
    f, l = smoke_train(
        args.index,
        args.labels,
        steps=args.steps,
        batch_size=args.batch_size,
        use_image=args.use_image,
        image_size=args.image_size,
        frame_window=args.frame_window,
        view_names=view_names,
        text_vocab_size=args.text_vocab_size,
        text_max_len=args.text_max_len,
    )
    print(f"smoke_train first_loss={f:.6f} last_loss={l:.6f}")


if __name__ == "__main__":
    main()
