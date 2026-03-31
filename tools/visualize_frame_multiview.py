#!/usr/bin/env python3
"""Visualize multi-view (4 cameras + radar) predictions for a selected frame."""

import glob
import os
import sys
import typing
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# add project root to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import get_infer_results
from ppdet.utils.check import (check_config, check_gpu, check_mlu, check_npu,
                               check_xpu)
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger
from ppdet.utils.visualizer import visualize_results
from ppdet.data.reader import Compose, BatchCompose


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 10
    RESAMPLE = Image.LANCZOS


logger = setup_logger('visualize_multiview')

CAMERA_DISPLAY_PATTERNS = {
    'front_rgb': ('Front', 0),
    'right_rgb': ('Right', 1),
    'back_rgb': ('Back', 2),
    'left_rgb': ('Left', 3),
}

# Order expected by the training pipeline when stacking camera tensors.
CANONICAL_CAMERA_SEQUENCE = ['back_rgb', 'front_rgb', 'left_rgb', 'right_rgb']


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the model weights (.pdparams) for inference.")
    parser.add_argument(
        "--im_id",
        type=int,
        default=None,
        help="COCO image id to visualize. Use this or --frame_token.")
    parser.add_argument(
        "--frame_token",
        type=str,
        default=None,
        help="Frame token substring (e.g. 0001) used to match the radar image filename.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multiview_visualizations",
        help="Directory to save composed visualization.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.4,
        help="Score threshold for drawing predictions on radar image.")
    parser.add_argument(
        "--camera_width",
        type=int,
        default=None,
        help="Optional width (in pixels) for each camera tile. Defaults to half radar width.")
    parser.add_argument(
        "--spacing",
        type=int,
        default=40,
        help="Vertical spacing (in pixels) between camera grid and radar image.")
    parser.add_argument(
        "--radar_image",
        type=str,
        default=None,
        help="Direct path to a radar image for inference (overrides dataset lookup).")
    parser.add_argument(
        "--camera_images",
        type=str,
        nargs='*',
        default=None,
        help="List of camera image paths corresponding to the radar frame.")
    parser.add_argument(
        "--radar_root",
        type=str,
        default=None,
        help="Directory that stores radar images when using --frame_token outside annotations.")
    parser.add_argument(
        "--camera_root",
        type=str,
        default=None,
        help="Directory that stores camera images when using --frame_token outside annotations.")
    parser.add_argument(
        "--custom_im_id",
        type=int,
        default=-1,
        help="Custom im_id to assign when using external images. Defaults to a hash of frame token.")
    return parser.parse_args()


def ensure_numpy(data):
    """Convert paddle.Tensor to numpy recursively."""
    if isinstance(data, paddle.Tensor):
        return data.numpy()
    if isinstance(data, typing.Sequence):
        return [ensure_numpy(d) for d in data]
    if isinstance(data, dict):
        return {k: ensure_numpy(v) for k, v in data.items()}
    return data


def prepare_output_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def build_asset_map(dataset):
    """Build mapping from im_id to radar/camera file paths."""
    asset_map = {}
    roidbs = getattr(dataset, 'roidbs', None)
    if not roidbs:
        return asset_map
    for rec in roidbs:
        im_id = rec.get('im_id')
        if im_id is None:
            continue
        if isinstance(im_id, np.ndarray):
            im_id_val = int(im_id.flatten()[0])
        elif isinstance(im_id, (list, tuple)):
            im_id_val = int(im_id[0])
        else:
            im_id_val = int(im_id)
        radar_path = rec.get('radar_im_file') or rec.get('im_file')
        camera_paths = rec.get('camera_im_file') or []
        if isinstance(camera_paths, str):
            camera_paths = [camera_paths]
        asset_map[im_id_val] = {
            'radar': radar_path,
            'cameras': list(camera_paths),
        }
    return asset_map


def extract_camera_tag(camera_path: str):
    """Extract camera tag (front_rgb, back_rgb, etc.) from filename."""
    basename = os.path.basename(camera_path).lower()
    for tag in CAMERA_DISPLAY_PATTERNS:
        if tag in basename:
            return tag
    return None


def infer_camera_label(camera_path: str):
    """Infer human-readable camera label and ordering from filename."""
    tag = extract_camera_tag(camera_path)
    if tag and tag in CAMERA_DISPLAY_PATTERNS:
        label, order = CAMERA_DISPLAY_PATTERNS[tag]
        return label, order
    return os.path.splitext(os.path.basename(camera_path))[0], 99


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Measure text width/height with backward compatibility."""
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, 'textsize'):
        return draw.textsize(text, font=font)
    # fallback
    font = font or ImageFont.load_default()
    return font.getsize(text)


def build_token_variants(frame_token: str):
    """Generate possible normalized variants for a frame token."""
    if not frame_token:
        return []
    token = frame_token.strip()
    variants = {token}
    if token.isdigit():
        variants.add(token.zfill(4))
        stripped = token.lstrip('0')
        variants.add(stripped if stripped else '0')
    if token.startswith('0'):
        stripped = token.lstrip('0')
        if stripped:
            variants.add(stripped)
    return [v for v in variants if v]


def resolve_radar_path(frame_token: str, radar_root: str):
    """Locate radar image using frame token within provided directory."""
    if not radar_root or not frame_token:
        return None
    variants = build_token_variants(frame_token)
    for variant in variants:
        candidate = os.path.join(radar_root,
                                 f"{variant}_cam_topview_bev_bw.png")
        if os.path.isfile(candidate):
            return candidate
    for variant in variants:
        pattern = os.path.join(radar_root, f"{variant}*")
        matches = sorted(glob.glob(pattern))
        for path in matches:
            if os.path.isfile(path):
                return path
    return None


def resolve_camera_paths(frame_token: str, camera_root: str):
    """Locate camera images using frame token within provided directory."""
    if not camera_root or not frame_token:
        return []
    variants = build_token_variants(frame_token)
    direction_tags = CANONICAL_CAMERA_SEQUENCE

    for variant in variants:
        candidates = []
        missing = False
        for tag in direction_tags:
            candidate = os.path.join(camera_root,
                                     f"{variant}_cam_{tag}.png")
            if not os.path.isfile(candidate):
                missing = True
                break
            candidates.append(candidate)
        if not missing:
            return candidates

    # fallback glob
    for variant in variants:
        pattern = os.path.join(camera_root, f"{variant}_cam_*.png")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches
    return []


def gather_manual_sample(args):
    """Aggregate manual paths provided directly or via directories."""
    if not (args.radar_image or args.frame_token):
        return None

    radar_path = args.radar_image
    if radar_path is None and args.frame_token and args.radar_root:
        radar_path = resolve_radar_path(args.frame_token, args.radar_root)

    camera_paths = []
    if args.camera_images:
        camera_paths = list(args.camera_images)
    elif args.frame_token and args.camera_root:
        camera_paths = resolve_camera_paths(args.frame_token,
                                            args.camera_root)

    unique_camera_paths = []
    seen = set()
    for path in camera_paths:
        if path and path not in seen and os.path.isfile(path):
            unique_camera_paths.append(path)
            seen.add(path)

    if not unique_camera_paths:
        raise FileNotFoundError(
            "No valid camera images found. Provide --camera_images or "
            "--camera_root with accessible files.")

    tagged_paths = []
    for idx, path in enumerate(unique_camera_paths):
        tag = extract_camera_tag(path)
        tagged_paths.append((path, tag, idx))

    def sort_key(item):
        path, tag, original_idx = item
        if tag in CANONICAL_CAMERA_SEQUENCE:
            return CANONICAL_CAMERA_SEQUENCE.index(tag), original_idx
        return len(CANONICAL_CAMERA_SEQUENCE), original_idx

    tagged_paths.sort(key=sort_key)
    ordered_camera_paths = [item[0] for item in tagged_paths]

    if not radar_path or not os.path.isfile(radar_path):
        raise FileNotFoundError(
            f"Radar image not found for the provided inputs: {radar_path}")

    inferred_token = args.frame_token
    if inferred_token is None:
        basename = os.path.basename(radar_path)
        inferred_token = basename.split('_')[0] if '_' in basename else basename

    if args.custom_im_id >= 0:
        im_id = int(args.custom_im_id)
    else:
        token_numeric = None
        if inferred_token and inferred_token.isdigit():
            token_numeric = int(inferred_token)
        im_id = token_numeric if token_numeric is not None else \
            abs(hash(radar_path)) % 100000000

    return {
        'im_id': im_id,
        'radar': radar_path,
        'cameras': ordered_camera_paths,
        'frame_token': inferred_token
    }


def create_eval_transforms(cfg, num_classes):
    """Create sample/batch transforms to mimic evaluation pipeline."""
    eval_reader = getattr(cfg, 'EvalReader', cfg.get('EvalReader', {}))
    sample_transforms = eval_reader.get('sample_transforms', [])
    batch_transforms = eval_reader.get('batch_transforms', [])
    collate_batch = eval_reader.get('collate_batch', True)
    sample_compose = Compose(sample_transforms, num_classes=num_classes)
    batch_compose = BatchCompose(
        batch_transforms,
        num_classes=num_classes,
        collate_batch=collate_batch)
    return sample_compose, batch_compose


def numpy_to_paddle(data):
    """Recursively convert numpy arrays to paddle.Tensor."""
    if isinstance(data, np.ndarray):
        return paddle.to_tensor(data)
    if isinstance(data, list):
        return [numpy_to_paddle(item) for item in data]
    if isinstance(data, tuple):
        return tuple(numpy_to_paddle(item) for item in data)
    if isinstance(data, dict):
        return {k: numpy_to_paddle(v) for k, v in data.items()}
    return data


def select_predictions_for_im_id(batch_res, target_im_id):
    """Filter batched inference results for a given im_id."""
    sample_res = {}
    for key in ['bbox', 'mask', 'segm', 'keypoint', 'pose3d']:
        entries = batch_res.get(key)
        if entries is None:
            continue
        if isinstance(entries, (list, tuple)):
            filtered = [
                item for item in entries
                if int(item.get('image_id', target_im_id)) == target_im_id
            ]
            sample_res[key] = filtered
        else:
            sample_res[key] = entries
    return sample_res


def infer_manual_sample(trainer, cfg, sample_entry, clsid2catid, num_classes):
    """Run inference on manually supplied radar/camera images."""
    sample_compose, batch_compose = create_eval_transforms(cfg, num_classes)
    im_id = int(sample_entry['im_id'])
    sample = {
        'im_id': np.array([im_id], dtype=np.int64),
        'radar_im_file': sample_entry['radar'],
        'camera_im_file': sample_entry['cameras'],
    }
    processed = sample_compose(sample)
    batch_np = batch_compose([processed])
    batch_paddle = numpy_to_paddle(batch_np)

    trainer.model.eval()
    with paddle.no_grad():
        outputs = trainer.model(batch_paddle)

    outs = ensure_numpy(outputs)
    outs['im_id'] = batch_np.get('im_id',
                                 np.array([[im_id]], dtype=np.int64))
    if 'bbox_num' in outs:
        outs['bbox_num'] = np.array(outs['bbox_num']).astype(
            np.int32).flatten()
    for key in ['im_shape', 'scale_factor']:
        if key in batch_np and key not in outs:
            outs[key] = batch_np[key]

    batch_res = get_infer_results(outs, clsid2catid)
    sample_res = select_predictions_for_im_id(batch_res, im_id)
    meta = {
        'im_id': im_id,
        'im_shape': batch_np.get('im_shape'),
        'scale_factor': batch_np.get('scale_factor'),
    }
    return sample_res, meta


def choose_target_im_id(asset_map, im_id=None, frame_token=None):
    if im_id is not None:
        if im_id not in asset_map:
            raise ValueError(f"im_id {im_id} not found in dataset.")
        return im_id
    if frame_token:
        candidates = []
        token_variants = build_token_variants(frame_token)
        for target_id, paths in asset_map.items():
            radar_path = paths.get('radar')
            if not radar_path:
                continue
            base = os.path.basename(radar_path)
            if any(variant and variant in base for variant in token_variants):
                candidates.append(target_id)
        candidates = sorted(set(candidates))
        if not candidates:
            raise ValueError(
                f"Failed to locate frame containing token '{frame_token}'.")
        if len(candidates) > 1:
            logger.warning(
                "Multiple frames matched token '%s'. Using the first match: %s.",
                frame_token, candidates)
        return candidates[0]
    raise ValueError("Please provide either --im_id or --frame_token.")


def gather_category_metadata(dataset):
    """Return mapping dictionaries for category ids -> class ids -> names."""
    clsid2catid = {}
    catid2name = {}

    if hasattr(dataset, 'catid2clsid'):
        clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
    if hasattr(dataset, 'cname2cid'):
        clsid2name = {v: k for k, v in dataset.cname2cid.items()}
        catid2name = {
            catid: clsid2name.get(clsid, str(catid))
            for clsid, catid in clsid2catid.items()
        }

    if not catid2name:
        anno_file = dataset.get_anno() if hasattr(dataset, 'get_anno') else None
        if anno_file:
            from ppdet.data.source.category import get_categories
            metric = getattr(dataset, 'metric', 'bbox')
            clsid2catid, catid2name = get_categories(metric, anno_file)
        else:
            catid2name = defaultdict(lambda: "unknown")

    return dict(clsid2catid), dict(catid2name)


def infer_single_frame(trainer, loader, target_im_id, clsid2catid):
    """Run inference and return predictions for the requested image id."""
    trainer.model.eval()
    with paddle.no_grad():
        for batch_data in loader:
            outputs = trainer.model(batch_data)
            reference = batch_data[0] if isinstance(batch_data,
                                                    typing.Sequence) else batch_data
            for key in ['im_shape', 'scale_factor', 'im_id', 'bbox_num']:
                if isinstance(reference, dict) and key in reference:
                    outputs[key] = reference[key]
            outputs = ensure_numpy(outputs)
            if 'im_id' not in outputs:
                continue

            im_ids = outputs['im_id']
            if isinstance(im_ids, np.ndarray):
                im_ids = im_ids.flatten().astype(np.int64)
            else:
                im_ids = np.array(im_ids, dtype=np.int64).flatten()
            if target_im_id not in im_ids:
                continue

            if 'bbox_num' in outputs:
                outputs['bbox_num'] = np.array(outputs['bbox_num']).astype(
                    np.int32).flatten()
            batch_res = get_infer_results(outputs, clsid2catid)

            sample_res = select_predictions_for_im_id(batch_res, target_im_id)
            meta = {'im_id': int(target_im_id)}
            im_shape = outputs.get('im_shape')
            scale_factor = outputs.get('scale_factor')
            match_indices = np.where(im_ids == target_im_id)[0]
            if match_indices.size > 0:
                idx = int(match_indices[0])
                if isinstance(im_shape, np.ndarray):
                    meta['im_shape'] = im_shape[idx]
                if isinstance(scale_factor, np.ndarray):
                    meta['scale_factor'] = scale_factor[idx]
            return sample_res, meta

    return None, None


def compose_multiview(camera_views,
                      radar_image,
                      spacing=40,
                      camera_tile_width=None,
                      background=(16, 16, 16)):
    """Create a composite image with 4 cameras (2x2) above radar image."""
    if not camera_views:
        return radar_image

    radar_w, radar_h = radar_image.size
    tile_width = camera_tile_width or max(radar_w // 2, 1)
    resized_views = []

    for label, img in camera_views:
        if tile_width <= 0:
            resized = img.copy()
        else:
            new_height = max(int(img.height * tile_width / max(img.width, 1)), 1)
            resized = img.resize((tile_width, new_height), RESAMPLE)
        resized_views.append((label, resized))

    rows = max((len(camera_views) + 1) // 2, 1)
    row_heights = []
    for row_idx in range(rows):
        row_imgs = [
            resized_views[i][1] for i in range(len(resized_views))
            if i // 2 == row_idx
        ]
        row_heights.append(max((img.height for img in row_imgs), default=1))

    cam_grid_width = tile_width * 2
    cam_grid_height = sum(row_heights)
    cam_canvas = Image.new('RGB', (cam_grid_width, cam_grid_height),
                           background)
    cam_draw = ImageDraw.Draw(cam_canvas)
    font = ImageFont.load_default()
    row_offsets = []
    offset = 0
    for h in row_heights:
        row_offsets.append(offset)
        offset += h

    for idx, (label, img) in enumerate(resized_views):
        row = idx // 2
        col = idx % 2
        row_height = row_heights[row]
        y_base = row_offsets[row]
        x = col * tile_width + max((tile_width - img.width) // 2, 0)
        y = y_base + max((row_height - img.height) // 2, 0)
        cam_canvas.paste(img, (x, y))
        if label:
            text_w, text_h = measure_text(cam_draw, label, font)
            text_x = x + 10
            text_y = y + 10
            cam_draw.rectangle(
                [text_x - 6, text_y - 4, text_x + text_w + 6, text_y + text_h + 4],
                fill=(0, 0, 0))
            cam_draw.text((text_x, text_y),
                          label,
                          font=font,
                          fill=(255, 255, 255))

    final_width = max(cam_canvas.width, radar_w)
    total_height = cam_canvas.height + spacing + radar_h
    composite = Image.new('RGB', (final_width, total_height), background)

    cam_x = (final_width - cam_canvas.width) // 2
    radar_x = (final_width - radar_w) // 2
    composite.paste(cam_canvas, (cam_x, 0))
    composite.paste(radar_image, (radar_x, cam_canvas.height + spacing))

    canvas_draw = ImageDraw.Draw(composite)
    radar_label = "Radar (Predictions)"
    label_w, label_h = measure_text(canvas_draw, radar_label, font)
    label_x = radar_x + 10
    label_y = cam_canvas.height + spacing + 10
    canvas_draw.rectangle(
        [label_x - 6, label_y - 4, label_x + label_w + 6, label_y + label_h + 4],
        fill=(0, 0, 0))
    canvas_draw.text((label_x, label_y),
                     radar_label,
                     font=font,
                     fill=(255, 255, 255))

    return composite


def main():
    args = parse_args()
    cfg = load_config(args.config)
    merge_args(cfg, args)
    merge_config(args.opt)

    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    if cfg.use_gpu:
        device = 'gpu'
    elif cfg.use_npu:
        device = 'npu'
    elif cfg.use_xpu:
        device = 'xpu'
    elif cfg.use_mlu:
        device = 'mlu'
    else:
        device = 'cpu'
    paddle.set_device(device)

    if args.weights:
        cfg.weights = args.weights
    if not getattr(cfg, 'weights', None):
        raise ValueError(
            "Model weights must be provided via config or --weights argument.")

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)

    if hasattr(cfg, 'weights') and not os.path.exists(cfg.weights):
        logger.warning("Weights file %s not found. Paddle will try to load it.",
                       cfg.weights)

    init_parallel_env()

    trainer = Trainer(cfg, mode='eval')
    trainer.load_weights(cfg.weights)
    dataset = trainer.dataset
    if hasattr(dataset, 'parse_dataset'):
        dataset.parse_dataset()
    loader = trainer.loader

    clsid2catid, catid2name = gather_category_metadata(dataset)
    cfg_num_classes = getattr(cfg, 'num_classes', None)
    if cfg_num_classes is None:
        try:
            cfg_num_classes = cfg.get('num_classes', None)
        except AttributeError:
            cfg_num_classes = None
    num_classes = getattr(dataset, 'num_classes', None)
    if not num_classes:
        if hasattr(dataset, 'cname2cid') and dataset.cname2cid:
            num_classes = len(dataset.cname2cid)
        elif cfg_num_classes:
            num_classes = int(cfg_num_classes)
        else:
            num_classes = max(len(catid2name), 1)

    manual_sample = None
    manual_mode = False
    if args.radar_image or args.camera_images:
        manual_sample = gather_manual_sample(args)
        manual_mode = True

    asset_map = {} if manual_mode else build_asset_map(dataset)
    target_im_id = None
    asset_entry = None

    if not manual_mode:
        if asset_map:
            try:
                target_im_id = choose_target_im_id(
                    asset_map, im_id=args.im_id, frame_token=args.frame_token)
                asset_entry = asset_map[target_im_id]
            except ValueError as exc:
                if args.frame_token and (args.radar_root or args.camera_root):
                    manual_sample = gather_manual_sample(args)
                    manual_mode = True
                else:
                    raise exc
        else:
            if args.frame_token or args.radar_image or args.camera_images:
                manual_sample = gather_manual_sample(args)
                manual_mode = True
            else:
                raise RuntimeError(
                    "Dataset does not provide radar/camera paths and no manual "
                    "inputs were supplied.")

    if manual_mode:
        predictions, meta = infer_manual_sample(
            trainer, cfg, manual_sample, clsid2catid, num_classes)
        target_im_id = manual_sample['im_id']
        radar_path = manual_sample['radar']
        camera_paths = manual_sample['cameras']
        frame_token_value = manual_sample.get('frame_token')
    else:
        radar_path = asset_entry.get('radar')
        camera_paths = asset_entry.get('cameras', [])
        predictions, meta = infer_single_frame(trainer, loader, target_im_id,
                                               clsid2catid)
        frame_token_value = args.frame_token

    if predictions is None:
        raise RuntimeError(
            f"Failed to obtain predictions for im_id {target_im_id}.")

    if not radar_path or not os.path.exists(radar_path):
        raise FileNotFoundError(
            f"Radar image not found for im_id={target_im_id}: {radar_path}")

    radar_image = Image.open(radar_path).convert('RGB')
    bbox_res = predictions.get('bbox')
    mask_res = predictions.get('mask')
    segm_res = predictions.get('segm')
    keypoint_res = predictions.get('keypoint')
    pose3d_res = predictions.get('pose3d')

    detections_pass = 0
    if bbox_res:
        detections_pass = sum(
            1 for det in bbox_res
            if det.get('score', 0.0) >= args.draw_threshold)

    radar_vis = visualize_results(
        radar_image.copy(), bbox_res, mask_res, segm_res, keypoint_res,
        pose3d_res, target_im_id, catid2name, threshold=args.draw_threshold)
    radar_vis = radar_vis.convert('RGB')

    camera_views = []
    for path in camera_paths:
        if not os.path.exists(path):
            logger.warning("Camera image missing: %s", path)
            continue
        tag = extract_camera_tag(path)
        order = CANONICAL_CAMERA_SEQUENCE.index(
            tag) if tag in CANONICAL_CAMERA_SEQUENCE else len(
                CANONICAL_CAMERA_SEQUENCE)
        label, display_order = infer_camera_label(path)
        camera_views.append((order, label, Image.open(path).convert('RGB')))

    camera_views.sort(key=lambda item: (item[0], item[1]))
    sorted_views = [(label, img) for _, label, img in camera_views]

    composite = compose_multiview(sorted_views,
                                  radar_vis,
                                  spacing=max(args.spacing, 0),
                                  camera_tile_width=args.camera_width)

    prepare_output_dir(args.output_dir)
    frame_token = frame_token_value or os.path.basename(radar_path).split(
        '_')[0]
    save_name = f"frame_{frame_token or target_im_id}.png"
    save_path = os.path.join(args.output_dir, save_name)
    composite.save(save_path, format='PNG')

    logger.info("Saved visualization to %s", save_path)
    logger.info("im_id=%s detections >= %.2f: %d", target_im_id,
                args.draw_threshold, detections_pass)
    if meta:
        logger.info("Metadata: %s", meta)


if __name__ == "__main__":
    main()
