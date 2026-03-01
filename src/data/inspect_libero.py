import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect one LIBERO HDF5 file and dump a few frames.")
    p.add_argument("--raw-dir", default="data/libero_raw")
    p.add_argument("--out-dir", default="viz/day1")
    return p.parse_args()


def save_ppm(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr[:, :, :3].tobytes())


def find_first_hdf5(raw_dir: Path) -> Optional[Path]:
    for file in sorted(raw_dir.rglob("*.hdf5")):
        return file
    for file in sorted(raw_dir.rglob("*.h5")):
        return file
    return None


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    file = find_first_hdf5(raw_dir)
    if file is None:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {raw_dir}")

    with h5py.File(file, "r") as f:
        data = f["data"]
        demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
        if not demo_keys:
            raise KeyError(f"No demo_* groups found in {file}")
        demo_key = demo_keys[0]
        demo = data[demo_key]
        problem_info_raw = data.attrs.get("problem_info", "{}")
        if isinstance(problem_info_raw, bytes):
            problem_info_raw = problem_info_raw.decode("utf-8")
        instruction = str(problem_info_raw)
        if "language_instruction" in instruction:
            # The attribute is JSON-like in LIBERO; printing raw is enough for probe.
            pass

        obs = demo["obs"]
        camera_name = "agentview_rgb" if "agentview_rgb" in obs else list(obs.keys())[0]
        imgs = obs[camera_name]
        n = int(imgs.shape[0])
        indices = sorted({0, max(0, n // 2), max(0, n - 1)})
        for idx in indices:
            save_ppm(out_dir / f"{file.stem}_{demo_key}_{idx}.ppm", np.array(imgs[idx]))

        action_shape = tuple(demo["actions"].shape) if "actions" in demo else ()
        low_dim_keys = sorted(list(obs.keys()))
        print(f"file={file}")
        print(f"demo={demo_key}")
        print(f"instruction={instruction}")
        print(f"camera_name={camera_name}")
        print(f"num_frames={n}")
        print(f"action_shape={action_shape}")
        print(f"obs_keys={low_dim_keys}")
        print(f"saved_frames={len(indices)}")


if __name__ == "__main__":
    main()
