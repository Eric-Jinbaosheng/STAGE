#!/usr/bin/env python3
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import mujoco
import numpy as np

TARGET_TO_SITE = {
    'black bowl': 'akita_black_bowl_1_default_site',
    'ramekin': 'glazed_rim_porcelain_ramekin_1_default_site',
    'cookie box': 'cookies_1_default_site',
    'top drawer': 'wooden_cabinet_1_top_region',
    'middle drawer': 'wooden_cabinet_1_middle_region',
}


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')


def clean_mujoco_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    for asset in root.findall('asset'):
        for child in list(asset):
            if child.tag in {'mesh', 'texture', 'material'}:
                asset.remove(child)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'geom':
                child.attrib.pop('material', None)
                child.attrib.pop('mesh', None)
                if child.get('type') == 'mesh':
                    parent.remove(child)
    return ET.tostring(root, encoding='unicode')


def split_obs_ptr(ptr: str) -> Tuple[str, str, int]:
    file_path, ref = ptr.split('#', 1)
    parts = ref.strip('/').split('/')
    if len(parts) < 5 or parts[0] != 'data':
        raise ValueError(f'Unsupported obs pointer: {ptr}')
    demo = '/'.join(parts[:2])
    frame = int(parts[-1])
    return file_path, demo, frame


class MujocoStateReader:
    def __init__(self):
        self.model_cache: Dict[Tuple[str, str], Tuple[mujoco.MjModel, str]] = {}

    def model_for(self, file_path: str, demo: str) -> mujoco.MjModel:
        key = (file_path, demo)
        if key in self.model_cache:
            return self.model_cache[key][0]
        with h5py.File(file_path, 'r') as f:
            xml = f[demo].attrs['model_file']
        model = mujoco.MjModel.from_xml_string(clean_mujoco_xml(xml))
        self.model_cache[key] = (model, xml)
        return model

    def read_positions(self, obs_ptr: str, targets: Tuple[str, str]) -> Dict[str, Any]:
        file_path, demo, frame = split_obs_ptr(obs_ptr)
        model = self.model_for(file_path, demo)
        data = mujoco.MjData(model)
        with h5py.File(file_path, 'r') as f:
            state = np.asarray(f[f'{demo}/states'][frame], dtype=float)
            ee_pos = np.asarray(f[f'{demo}/obs/ee_pos'][frame], dtype=float)
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq : model.nq + model.nv]
        mujoco.mj_forward(model, data)

        out: Dict[str, Any] = {'ee_pos': ee_pos.tolist(), 'position_source': obs_ptr}
        for label, target in [('original', targets[0]), ('counterfactual', targets[1])]:
            site_name = TARGET_TO_SITE.get(str(target or '').lower())
            pos = None
            if site_name:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if sid >= 0:
                    pos = data.site_xpos[sid].copy()
            out[f'{label}_target_site'] = site_name
            out[f'{label}_target_pos'] = pos.tolist() if pos is not None else None
        return out


def angle_deg(v1, v2) -> Optional[float]:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return None
    cos = np.clip(float(np.dot(v1, v2) / denom), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def add_geometry(row: Dict[str, Any], reader: MujocoStateReader) -> Dict[str, Any]:
    ptr = row.get('image_source') or row.get('obs_ptr')
    if not ptr or '#' not in str(ptr):
        row['position_extraction_status'] = 'missing_obs_ptr'
        return row
    try:
        pos = reader.read_positions(str(ptr), (row.get('original_target_object'), row.get('counterfactual_target_object')))
        row.update(pos)
        p_orig = row.get('original_target_pos')
        p_cf = row.get('counterfactual_target_pos')
        ee = row.get('ee_pos')
        if p_orig is not None and p_cf is not None and ee is not None:
            p_orig_np = np.asarray(p_orig, dtype=float)
            p_cf_np = np.asarray(p_cf, dtype=float)
            ee_np = np.asarray(ee, dtype=float)
            row['target_distance'] = float(np.linalg.norm(p_orig_np - p_cf_np))
            row['target_angle_deg'] = angle_deg(p_orig_np - ee_np, p_cf_np - ee_np)
            row['position_extraction_status'] = 'ok'
        else:
            row['target_distance'] = None
            row['target_angle_deg'] = None
            row['position_extraction_status'] = 'missing_target_site'
    except Exception as exc:
        row['position_extraction_status'] = f'error:{type(exc).__name__}:{exc}'
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='Add LIBERO MuJoCo 3D target positions to Exp2 predictions.')
    ap.add_argument('--predictions', default='outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    pred_path = Path(args.predictions)
    out_path = Path(args.out) if args.out else pred_path.parent / 'analysis' / 'predictions_with_positions.jsonl'
    reader = MujocoStateReader()
    rows = [add_geometry(row, reader) for row in read_jsonl(pred_path)]
    write_jsonl(out_path, rows)
    status = {}
    for row in rows:
        status[row.get('position_extraction_status')] = status.get(row.get('position_extraction_status'), 0) + 1
    print(json.dumps({'out': str(out_path), 'num_rows': len(rows), 'status': status}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
