from __future__ import annotations

"""
Radar json -> 单通道 PPI(log_count) 离线批处理脚本。

功能：
1) 递归处理 dataset/<split> 下所有场景与塔目录
2) 输入：<scene>/<tower>/radar/radar_XXXXXX.json（仅使用 range/azimuth/elevation）
3) 输出：<scene>/<tower>/ppi_npy/ppi_XXXXXX.npy 与 <scene>/<tower>/ppi_png/ppi_XXXXXX.png
4) 参数固定来自 postprocess_config.py，并将本次参数同步写入两个输出目录的 ppi_params.yaml
5) apply_correction 可选：基于 gt_sensor 的 tower_pose_neu + radar_extrinsics_neu 去除 roll/pitch
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import postprocess_config as cfg


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quat_to_R(q: Dict[str, float]) -> np.ndarray:
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_keep_yaw_only(R: np.ndarray) -> np.ndarray:
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def parse_detections_xyz(radar_json: Dict) -> np.ndarray:
    dets_xyz: List[Tuple[float, float, float]] = []
    for det in radar_json.get("detections", []):
        r = float(det.get("range", 0.0))
        az = float(det.get("azimuth", 0.0))
        if abs(az) > math.pi * 2:
            az = math.radians(az)
        el = float(det.get("elevation", 0.0))
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)
        dets_xyz.append((x, y, z))
    if len(dets_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(dets_xyz, dtype=np.float32)


def load_pcd_xyz(path: Path) -> np.ndarray:
    fields = None
    data_idx = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
            continue
        if line.startswith("DATA"):
            parts = line.split()
            if len(parts) < 2 or parts[1].lower() != "ascii":
                raise ValueError(f"Only ascii PCD is supported: {path}")
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError(f"PCD missing DATA section: {path}")
    if not fields:
        fields = ["x", "y", "z"]

    field_index = {name: idx for idx, name in enumerate(fields)}
    for name in ("x", "y", "z"):
        if name not in field_index:
            raise ValueError(f"PCD missing field '{name}': {path}")

    max_idx = max(field_index["x"], field_index["y"], field_index["z"])
    points = []
    for raw in lines[data_idx:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) <= max_idx:
            continue
        points.append(
            (
                float(parts[field_index["x"]]),
                float(parts[field_index["y"]]),
                float(parts[field_index["z"]]),
            )
        )
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


@dataclass
class CorrectionTransform:
    R_radar: np.ndarray
    t_radar: np.ndarray
    delta: np.ndarray


def build_correction_transform(gt_sensor_data: Dict, tower: str) -> Optional[CorrectionTransform]:
    """
    中文备注：和 translation_radar2pcd.py 保持同一校正逻辑：
    radar点 -> 塔坐标(NEU) -> 去除roll/pitch(仅保留yaw)。
    """
    tower_data = gt_sensor_data.get("towers", {}).get(tower, {})
    tower_pose_neu = tower_data.get("tower_pose_neu")
    radar_ext_neu = tower_data.get("radar_extrinsics_neu")
    if not tower_pose_neu or not radar_ext_neu:
        return None

    R_tower = quat_to_R(tower_pose_neu["rotation_quat_neu"])
    R_tower_yaw = R_keep_yaw_only(R_tower)
    delta = R_tower_yaw @ R_tower.T

    t_radar = np.array(
        [
            radar_ext_neu["t_neu"]["x"],
            radar_ext_neu["t_neu"]["y"],
            radar_ext_neu["t_neu"]["z"],
        ],
        dtype=np.float64,
    )
    R_radar = quat_to_R(radar_ext_neu["rotation_quat_neu"])
    return CorrectionTransform(R_radar=R_radar, t_radar=t_radar, delta=delta)


def apply_correction_xyz(points_radar_xyz: np.ndarray, corr: CorrectionTransform) -> np.ndarray:
    if points_radar_xyz.size == 0:
        return points_radar_xyz
    pts = (corr.R_radar @ points_radar_xyz.T) + corr.t_radar.reshape(3, 1)
    pts = corr.delta @ pts
    return pts.T.astype(np.float32)


def build_gaussian_kernel_1d(sigma: float, kernel_size: int = 0) -> np.ndarray:
    sigma = float(max(1e-6, sigma))
    if kernel_size <= 0:
        kernel_size = int(max(3, round(6.0 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    kernel = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
    kernel = kernel / max(float(kernel.sum()), 1e-12)
    return kernel.astype(np.float32)


def conv1d_axis(arr: np.ndarray, kernel: np.ndarray, axis: int, pad_mode: str) -> np.ndarray:
    """
    中文备注：纯 numpy 的一维卷积（沿某一轴），避免依赖 torch/scipy。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    pad = kernel.shape[0] // 2
    if axis == 1:
        mode = "wrap" if pad_mode == "circular" else "edge"
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode=mode)
        windows = sliding_window_view(padded, window_shape=kernel.shape[0], axis=1)
        out = np.tensordot(windows, kernel, axes=([2], [0]))
        return out.astype(np.float32)

    mode = "edge"
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode=mode)
    windows = sliding_window_view(padded, window_shape=kernel.shape[0], axis=0)
    out = np.tensordot(windows, kernel, axes=([2], [0]))
    return out.astype(np.float32)


def format_yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = f"{float(v):.10g}"
        return s
    if v is None:
        return "null"
    text = str(v)
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", " "]):
        return f"\"{text}\""
    return text


def dump_yaml_like(data, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(dump_yaml_like(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {format_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for x in data:
            if isinstance(x, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(dump_yaml_like(x, indent + 1))
            else:
                lines.append(f"{sp}- {format_yaml_scalar(x)}")
        return "\n".join(lines)
    return f"{sp}{format_yaml_scalar(data)}"


@dataclass
class PpiParams:
    r_min: float
    r_max: float
    dr: float
    theta_min_deg: float
    theta_max_deg: float
    dtheta_deg: float
    cart_size: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    scan_convert_mode: str
    log_count_apply_range_gain: bool
    range_gain_mode: str
    range_power: float
    ref_range: float
    min_range: float
    min_gain: float
    max_gain: float
    physical_model_enabled: bool
    az_sigma_bins: float
    range_sigma_bins: float
    az_kernel_size: int
    range_kernel_size: int
    log_count_clip_max: float
    log_count_normalize: bool
    png_percentile: float
    apply_correction: bool


def build_params_from_cfg(apply_correction: bool) -> PpiParams:
    return PpiParams(
        r_min=float(cfg.DEFAULT_PPI_R_MIN),
        r_max=float(cfg.DEFAULT_PPI_R_MAX),
        dr=float(cfg.DEFAULT_PPI_DR),
        theta_min_deg=float(cfg.DEFAULT_PPI_THETA_MIN_DEG),
        theta_max_deg=float(cfg.DEFAULT_PPI_THETA_MAX_DEG),
        dtheta_deg=float(cfg.DEFAULT_PPI_DTHETA_DEG),
        cart_size=int(cfg.DEFAULT_PPI_CART_SIZE),
        x_min=float(cfg.DEFAULT_PPI_X_MIN),
        x_max=float(cfg.DEFAULT_PPI_X_MAX),
        y_min=float(cfg.DEFAULT_PPI_Y_MIN),
        y_max=float(cfg.DEFAULT_PPI_Y_MAX),
        scan_convert_mode=str(cfg.DEFAULT_PPI_SCAN_CONVERT_MODE).lower(),
        log_count_apply_range_gain=bool(cfg.DEFAULT_PPI_LOG_COUNT_APPLY_RANGE_GAIN),
        range_gain_mode=str(cfg.DEFAULT_PPI_RANGE_GAIN_MODE).lower(),
        range_power=float(cfg.DEFAULT_PPI_RANGE_POWER),
        ref_range=float(cfg.DEFAULT_PPI_REF_RANGE),
        min_range=float(cfg.DEFAULT_PPI_MIN_RANGE),
        min_gain=float(cfg.DEFAULT_PPI_MIN_GAIN),
        max_gain=float(cfg.DEFAULT_PPI_MAX_GAIN),
        physical_model_enabled=bool(cfg.DEFAULT_PPI_PHYSICAL_MODEL_ENABLED),
        az_sigma_bins=float(cfg.DEFAULT_PPI_AZ_SIGMA_BINS),
        range_sigma_bins=float(cfg.DEFAULT_PPI_RANGE_SIGMA_BINS),
        az_kernel_size=int(cfg.DEFAULT_PPI_AZ_KERNEL_SIZE),
        range_kernel_size=int(cfg.DEFAULT_PPI_RANGE_KERNEL_SIZE),
        log_count_clip_max=float(cfg.DEFAULT_PPI_LOG_COUNT_CLIP_MAX),
        log_count_normalize=bool(cfg.DEFAULT_PPI_LOG_COUNT_NORMALIZE),
        png_percentile=float(cfg.DEFAULT_PPI_PNG_PERCENTILE),
        apply_correction=bool(apply_correction),
    )


def compute_range_gain(ranges: np.ndarray, p: PpiParams) -> np.ndarray:
    safe_r = np.maximum(ranges.astype(np.float32), max(1e-6, p.min_range))
    ref_r = max(p.ref_range, p.min_range)
    if p.range_gain_mode == "attenuate":
        gain = (ref_r / safe_r) ** p.range_power
    else:
        gain = (safe_r / ref_r) ** p.range_power
    gain = np.clip(gain, p.min_gain, p.max_gain)
    return gain.astype(np.float32)


def clip_norm_log_count(x: np.ndarray, p: PpiParams) -> np.ndarray:
    out = np.maximum(x.astype(np.float32), 0.0)
    if p.log_count_clip_max > 0:
        out = np.clip(out, 0.0, p.log_count_clip_max)
        if p.log_count_normalize:
            out = out / max(p.log_count_clip_max, 1e-6)
    return out.astype(np.float32)


def prepare_scan_lookup(p: PpiParams):
    h = int(p.cart_size)
    w = int(p.cart_size)
    dx = (p.x_max - p.x_min) / max(float(w), 1e-6)
    dy = (p.y_max - p.y_min) / max(float(h), 1e-6)

    y_centers = p.y_min + (np.arange(h, dtype=np.float32) + 0.5) * dy
    x_centers = p.x_min + (np.arange(w, dtype=np.float32) + 0.5) * dx
    xx, yy = np.meshgrid(x_centers, y_centers)
    rr = np.sqrt(xx * xx + yy * yy)
    tt = np.arctan2(yy, xx)

    theta_min = math.radians(p.theta_min_deg)
    theta_max = math.radians(p.theta_max_deg)
    dtheta = math.radians(max(p.dtheta_deg, 1e-6))

    rf = (rr - p.r_min) / max(p.dr, 1e-6)
    tf = (tt - theta_min) / max(dtheta, 1e-6)
    valid = np.logical_and(rr >= p.r_min, rr < p.r_max)
    valid = np.logical_and(valid, np.logical_and(tt >= theta_min, tt < theta_max))

    full_azimuth_span = abs((theta_max - theta_min) - 2 * math.pi) < 1e-3
    return rf.astype(np.float32), tf.astype(np.float32), valid, full_azimuth_span


def scan_convert_polar_to_cartesian(
    polar_map: np.ndarray,
    rf: np.ndarray,
    tf: np.ndarray,
    valid_in: np.ndarray,
    mode: str,
    full_azimuth_span: bool,
) -> np.ndarray:
    h_p, w_p = polar_map.shape
    valid = valid_in.copy()
    out = np.zeros_like(rf, dtype=np.float32)

    if mode == "nearest":
        ir = np.rint(rf).astype(np.int32)
        it = np.rint(tf).astype(np.int32)
        valid = np.logical_and(valid, np.logical_and(ir >= 0, ir < h_p))
        if full_azimuth_span:
            it = np.mod(it, w_p)
        else:
            valid = np.logical_and(valid, np.logical_and(it >= 0, it < w_p))
            it = np.clip(it, 0, w_p - 1)
        ir = np.clip(ir, 0, h_p - 1)
        out[valid] = polar_map[ir[valid], it[valid]]
        return out

    r0 = np.floor(rf).astype(np.int32)
    r1 = r0 + 1
    t0 = np.floor(tf).astype(np.int32)
    t1 = t0 + 1
    wr = rf - r0
    wt = tf - t0

    valid = np.logical_and(valid, np.logical_and(r0 >= 0, r1 < h_p))
    if full_azimuth_span:
        t0 = np.mod(t0, w_p)
        t1 = np.mod(t1, w_p)
    else:
        valid = np.logical_and(valid, np.logical_and(t0 >= 0, t1 < w_p))
        t0 = np.clip(t0, 0, w_p - 1)
        t1 = np.clip(t1, 0, w_p - 1)

    r0s = np.clip(r0, 0, h_p - 1)
    r1s = np.clip(r1, 0, h_p - 1)
    t0s = np.clip(t0, 0, w_p - 1)
    t1s = np.clip(t1, 0, w_p - 1)

    v00 = polar_map[r0s, t0s]
    v01 = polar_map[r0s, t1s]
    v10 = polar_map[r1s, t0s]
    v11 = polar_map[r1s, t1s]
    out_val = (1 - wr) * (1 - wt) * v00 + (1 - wr) * wt * v01 + wr * (1 - wt) * v10 + wr * wt * v11
    out[valid] = out_val[valid]
    return out.astype(np.float32)


def build_log_count_map(points_xyz: np.ndarray, p: PpiParams, rf, tf, valid_lookup, full_azimuth_span) -> np.ndarray:
    """
    中文备注：流程保持一致：
    点云几何 -> 极坐标栅格 count -> log1p -> polar物理卷积 -> scan conversion -> cart log_count。
    """
    theta_min = math.radians(p.theta_min_deg)
    theta_max = math.radians(p.theta_max_deg)
    dtheta = math.radians(max(p.dtheta_deg, 1e-6))

    h_p = int(math.ceil((p.r_max - p.r_min) / max(p.dr, 1e-6)))
    w_p = int(math.ceil((theta_max - theta_min) / max(dtheta, 1e-6)))
    count_map = np.zeros((h_p, w_p), dtype=np.float32)

    if points_xyz.shape[0] > 0:
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arctan2(y, x)

        valid = np.logical_and(r >= p.r_min, r < p.r_max)
        valid = np.logical_and(valid, np.logical_and(theta >= theta_min, theta < theta_max))
        if np.any(valid):
            r = r[valid]
            theta = theta[valid]
            ir = ((r - p.r_min) / p.dr).astype(np.int32)
            it = ((theta - theta_min) / dtheta).astype(np.int32)
            ir = np.clip(ir, 0, h_p - 1)
            it = np.clip(it, 0, w_p - 1)

            if p.log_count_apply_range_gain:
                weight = compute_range_gain(r, p)
            else:
                weight = np.ones_like(r, dtype=np.float32)
            np.add.at(count_map, (ir, it), weight)

    log_count = np.log1p(np.maximum(count_map, 0.0)).astype(np.float32)
    log_count = clip_norm_log_count(log_count, p)

    if p.physical_model_enabled:
        if p.az_sigma_bins > 0:
            k_az = build_gaussian_kernel_1d(p.az_sigma_bins, p.az_kernel_size)
            az_pad_mode = "circular" if full_azimuth_span else "edge"
            log_count = conv1d_axis(log_count, k_az, axis=1, pad_mode=az_pad_mode)
        if p.range_sigma_bins > 0:
            k_rg = build_gaussian_kernel_1d(p.range_sigma_bins, p.range_kernel_size)
            log_count = conv1d_axis(log_count, k_rg, axis=0, pad_mode="edge")

    cart = scan_convert_polar_to_cartesian(
        polar_map=log_count,
        rf=rf,
        tf=tf,
        valid_in=valid_lookup,
        mode=p.scan_convert_mode,
        full_azimuth_span=full_azimuth_span,
    )
    cart = clip_norm_log_count(cart, p)
    return cart.astype(np.float32)


def log_count_to_png_uint8(log_count: np.ndarray, p: PpiParams) -> np.ndarray:
    x = np.maximum(log_count.astype(np.float32), 0.0)
    if p.log_count_normalize:
        y = np.clip(x, 0.0, 1.0)
        return np.round(y * 255.0).astype(np.uint8)

    if p.log_count_clip_max > 0:
        y = np.clip(x / max(p.log_count_clip_max, 1e-6), 0.0, 1.0)
        return np.round(y * 255.0).astype(np.uint8)

    vmax = float(np.percentile(x, p.png_percentile)) if np.any(x > 0) else 1.0
    vmax = max(vmax, 1e-6)
    y = np.clip(x / vmax, 0.0, 1.0)
    return np.round(y * 255.0).astype(np.uint8)


def write_params_yaml(path: Path, p: PpiParams, scene: str, tower: str, split: str):
    data = {
        "meta": {
            "scene": scene,
            "tower": tower,
            "split": split,
            "format": "single_channel_log_count",
            "apply_correction": bool(p.apply_correction),
        },
        "polar_grid": {
            "r_min": p.r_min,
            "r_max": p.r_max,
            "dr": p.dr,
            "theta_min_deg": p.theta_min_deg,
            "theta_max_deg": p.theta_max_deg,
            "dtheta_deg": p.dtheta_deg,
        },
        "cartesian_grid": {
            "size": p.cart_size,
            "x_min": p.x_min,
            "x_max": p.x_max,
            "y_min": p.y_min,
            "y_max": p.y_max,
            "scan_convert_mode": p.scan_convert_mode,
        },
        "range_gain": {
            "enabled": bool(p.log_count_apply_range_gain),
            "mode": p.range_gain_mode,
            "range_power": p.range_power,
            "ref_range": p.ref_range,
            "min_range": p.min_range,
            "min_gain": p.min_gain,
            "max_gain": p.max_gain,
        },
        "physical_model": {
            "enabled": bool(p.physical_model_enabled),
            "az_sigma_bins": p.az_sigma_bins,
            "range_sigma_bins": p.range_sigma_bins,
            "az_kernel_size": p.az_kernel_size,
            "range_kernel_size": p.range_kernel_size,
        },
        "log_count_post": {
            "clip_max": p.log_count_clip_max,
            "normalize": bool(p.log_count_normalize),
            "png_percentile": p.png_percentile,
        },
    }
    text = dump_yaml_like(data) + "\n"
    path.write_text(text, encoding="utf-8")


def iter_scene_dirs(split_root: Path) -> Iterable[Path]:
    for p in sorted(split_root.iterdir()):
        if p.is_dir():
            yield p


def iter_tower_dirs(scene_dir: Path) -> Iterable[Path]:
    for p in sorted(scene_dir.iterdir()):
        if p.is_dir() and p.name.startswith("CoastGuard"):
            yield p


def extract_tick_str(radar_json_path: Path) -> Optional[str]:
    stem = radar_json_path.stem
    if "_" not in stem:
        return None
    suffix = stem.split("_")[-1]
    return suffix if suffix.isdigit() else None


def process_pcd_file(
    pcd_path: Path,
    out_npy: Path,
    out_png: Path,
    p: PpiParams,
    gt_sensor_path: Optional[Path] = None,
    tower: Optional[str] = None,
    lookup=None,
    overwrite: bool = False,
) -> bool:
    if (not overwrite) and out_npy.exists() and out_png.exists():
        return False

    points_xyz = load_pcd_xyz(pcd_path)
    if p.apply_correction and gt_sensor_path is not None and tower is not None and gt_sensor_path.is_file():
        gt_sensor_data = load_json(gt_sensor_path)
        corr = build_correction_transform(gt_sensor_data, tower)
        if corr is not None:
            points_xyz = apply_correction_xyz(points_xyz, corr)

    if lookup is None:
        lookup = prepare_scan_lookup(p)
    rf, tf, valid_lookup, full_azimuth_span = lookup

    log_count = build_log_count_map(points_xyz, p, rf, tf, valid_lookup, full_azimuth_span)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, log_count.astype(np.float32))

    img = log_count_to_png_uint8(log_count, p)
    Image.fromarray(img, mode="L").save(out_png)
    return True


def process_pcd_tick(
    run_dir: Path,
    tick: int,
    tower: str,
    p: PpiParams,
    overwrite: bool = False,
    lookup=None,
    npy_dirname: str = cfg.DEFAULT_PPI_NPY_DIRNAME,
    png_dirname: str = cfg.DEFAULT_PPI_PNG_DIRNAME,
) -> bool:
    tick_str = f"{tick:06d}"
    pcd_path = run_dir / tower / "radar_pcd" / f"radar_{tick_str}.pcd"
    if not pcd_path.is_file():
        return False

    out_npy_dir = run_dir / tower / str(npy_dirname)
    out_png_dir = run_dir / tower / str(png_dirname)
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    param_name = cfg.DEFAULT_PPI_PARAM_BASENAME
    write_params_yaml(out_npy_dir / param_name, p, run_dir.name, tower, run_dir.parent.name)
    write_params_yaml(out_png_dir / param_name, p, run_dir.name, tower, run_dir.parent.name)

    gt_sensor_path = run_dir / "gt_sensor" / f"gt_{tick_str}_tosensor.json"
    return process_pcd_file(
        pcd_path=pcd_path,
        out_npy=out_npy_dir / f"ppi_{tick_str}.npy",
        out_png=out_png_dir / f"ppi_{tick_str}.png",
        p=p,
        gt_sensor_path=gt_sensor_path,
        tower=tower,
        lookup=lookup,
        overwrite=overwrite,
    )


def process_split_recursively(dataset_root: Path,
                              split: str,
                              p: PpiParams,
                              overwrite: bool,
                              max_files: int = -1,
                              npy_dirname: str = cfg.DEFAULT_PPI_NPY_DIRNAME,
                              png_dirname: str = cfg.DEFAULT_PPI_PNG_DIRNAME) -> Tuple[int, int]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")

    rf = tf = valid_lookup = full_azimuth_span = None
    rf, tf, valid_lookup, full_azimuth_span = prepare_scan_lookup(p)

    processed = 0
    skipped = 0
    print(f"[PPI] split={split} root={split_root}")

    for scene_dir in iter_scene_dirs(split_root):
        gt_sensor_dir = scene_dir / "gt_sensor"
        if not gt_sensor_dir.is_dir():
            print(f"[Warn] scene has no gt_sensor, skip correction support: {scene_dir}")

        for tower_dir in iter_tower_dirs(scene_dir):
            radar_dir = tower_dir / "radar"
            if not radar_dir.is_dir():
                continue

            out_npy_dir = tower_dir / str(npy_dirname)
            out_png_dir = tower_dir / str(png_dirname)
            out_npy_dir.mkdir(parents=True, exist_ok=True)
            out_png_dir.mkdir(parents=True, exist_ok=True)

            param_name = cfg.DEFAULT_PPI_PARAM_BASENAME
            write_params_yaml(out_npy_dir / param_name, p, scene_dir.name, tower_dir.name, split)
            write_params_yaml(out_png_dir / param_name, p, scene_dir.name, tower_dir.name, split)

            radar_files = sorted(radar_dir.glob("radar_*.json"))
            if len(radar_files) == 0:
                continue

            for radar_json_path in radar_files:
                if int(max_files) > 0 and processed >= int(max_files):
                    return processed, skipped
                tick_str = extract_tick_str(radar_json_path)
                if tick_str is None:
                    skipped += 1
                    continue

                out_npy = out_npy_dir / f"ppi_{tick_str}.npy"
                out_png = out_png_dir / f"ppi_{tick_str}.png"
                if (not overwrite) and out_npy.exists() and out_png.exists():
                    skipped += 1
                    continue

                radar_data = load_json(radar_json_path)
                points_xyz = parse_detections_xyz(radar_data)

                if p.apply_correction:
                    gt_sensor_path = gt_sensor_dir / f"gt_{tick_str}_tosensor.json"
                    if gt_sensor_path.is_file():
                        gt_sensor_data = load_json(gt_sensor_path)
                        corr = build_correction_transform(gt_sensor_data, tower_dir.name)
                        if corr is not None:
                            points_xyz = apply_correction_xyz(points_xyz, corr)
                    # 中文备注：校正开启但缺失 gt_sensor 时，自动回退为不校正，不中断流程。

                log_count = build_log_count_map(points_xyz, p, rf, tf, valid_lookup, full_azimuth_span)
                np.save(out_npy, log_count.astype(np.float32))

                img = log_count_to_png_uint8(log_count, p)
                Image.fromarray(img, mode="L").save(out_png)

                processed += 1

        print(f"[PPI] scene={scene_dir.name} done, processed={processed}, skipped={skipped}")

    return processed, skipped


def main():
    ap = argparse.ArgumentParser(description="Batch convert radar json to single-channel PPI(log_count).")
    ap.add_argument("--dataset-root", type=Path, default=cfg.DEFAULT_PPI_DATASET_ROOT,
                    help="Dataset root directory, e.g. .../sealand_data/dataset")
    ap.add_argument("--split", type=str, default=cfg.DEFAULT_PPI_SPLIT,
                    help="Split name under dataset root, e.g. Train/Valid/Test")
    ap.add_argument("--apply-correction", type=str2bool, default=cfg.DEFAULT_PPI_APPLY_CORRECTION,
                    help="Apply tower roll/pitch correction using gt_sensor (true/false)")
    ap.add_argument("--overwrite", type=str2bool, default=cfg.DEFAULT_PPI_OVERWRITE,
                    help="Overwrite existing ppi files (true/false)")
    ap.add_argument("--max-files", type=int, default=-1,
                    help="Debug option: process at most N radar json files; -1 means all")
    ap.add_argument("--npy-dirname", type=str, default=cfg.DEFAULT_PPI_NPY_DIRNAME,
                    help="Output directory name under tower for npy files")
    ap.add_argument("--png-dirname", type=str, default=cfg.DEFAULT_PPI_PNG_DIRNAME,
                    help="Output directory name under tower for png files")
    ap.add_argument("--range-power", type=float, default=None,
                    help="Override range_power for distance gain")
    ap.add_argument("--ref-range", type=float, default=None,
                    help="Override ref_range for distance gain")
    ap.add_argument("--min-gain", type=float, default=None,
                    help="Override min_gain for distance gain")
    ap.add_argument("--log-count-apply-range-gain", type=str2bool, default=None,
                    help="Override whether log_count applies range gain")
    args = ap.parse_args()

    p = build_params_from_cfg(apply_correction=bool(args.apply_correction))
    if args.range_power is not None:
        p.range_power = float(args.range_power)
    if args.ref_range is not None:
        p.ref_range = float(args.ref_range)
    if args.min_gain is not None:
        p.min_gain = float(args.min_gain)
    if args.log_count_apply_range_gain is not None:
        p.log_count_apply_range_gain = bool(args.log_count_apply_range_gain)

    processed, skipped = process_split_recursively(
        dataset_root=args.dataset_root,
        split=args.split,
        p=p,
        overwrite=bool(args.overwrite),
        max_files=int(args.max_files),
        npy_dirname=str(args.npy_dirname),
        png_dirname=str(args.png_dirname),
    )
    print(f"[PPI] completed. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
