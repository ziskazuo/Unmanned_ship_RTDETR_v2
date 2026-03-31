#!/usr/bin/env python3
# Copyright (c) 2024
#
# Visualize evaluation predictions for the trained Radar-Camera RT-DETR model.

import os
import sys
import typing
import numpy as np
from tqdm import tqdm
from PIL import Image

# add project root to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.utils.logger import setup_logger
from ppdet.utils.check import (check_config, check_gpu, check_npu, check_xpu,
                               check_mlu)
from ppdet.metrics.coco_utils import get_infer_results
from ppdet.data.source.category import get_categories
from ppdet.utils.visualizer import visualize_results


logger = setup_logger('visualize_eval')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        default="visualize_eval",
        type=str,
        help="Directory to save visualized predictions.")
    parser.add_argument(
        "--draw_threshold",
        default=0.5,
        type=float,
        help="Score threshold for drawing predictions.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the model weights (.pdparams) for inference.")
    parser.add_argument(
        "--max_images",
        default=-1,
        type=int,
        help="Maximum number of validation samples to visualize (-1 for all).")
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="Skip saving images without valid predictions.")
    return parser.parse_args()


def build_imid_path_map(dataset):
    """Build mapping from image id to radar image path."""
    imid2path = {}
    if getattr(dataset, 'roidbs', None) is None:
        return imid2path
    for rec in dataset.roidbs:
        im_id = rec.get('im_id', None)
        if im_id is None:
            continue
        if isinstance(im_id, np.ndarray):
            key = int(im_id[0])
        elif isinstance(im_id, (list, tuple)):
            key = int(im_id[0])
        else:
            key = int(im_id)
        radar_path = rec.get('radar_im_file') or rec.get('im_file')
        if radar_path is not None:
            imid2path[key] = radar_path
    return imid2path


def ensure_numpy(data):
    """Convert paddle.Tensor to numpy array recursively."""
    if isinstance(data, paddle.Tensor):
        return data.numpy()
    if isinstance(data, typing.Sequence):
        return [ensure_numpy(d) for d in data]
    if isinstance(data, dict):
        return {k: ensure_numpy(v) for k, v in data.items()}
    return data


def prepare_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_predictions(cfg, args):
    # init parallel env
    init_parallel_env()

    trainer = Trainer(cfg, mode='eval')
    trainer.load_weights(cfg.weights)

    dataset = trainer.dataset
    loader = trainer.loader

    imid2path = build_imid_path_map(dataset)

    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()} \
        if hasattr(dataset, 'catid2clsid') else {}
    if not clsid2catid:
        # fallback using annotation file
        anno_file = dataset.get_anno()
        clsid2catid, _ = get_categories(cfg.metric, anno_file=anno_file)
    if hasattr(dataset, 'catid2clsid') and hasattr(dataset, 'cname2cid'):
        clsid2name = {v: k for k, v in dataset.cname2cid.items()}
        catid2name = {
            catid: clsid2name.get(clsid, str(catid))
            for catid, clsid in dataset.catid2clsid.items()
        }
    else:
        anno_file = dataset.get_anno()
        _, catid2name = get_categories(cfg.metric, anno_file=anno_file)

    prepare_output_dir(args.output_dir)
    trainer.model.eval()

    saved_images = 0

    with paddle.no_grad():
        for step_id, data in enumerate(tqdm(loader)):
            outs = trainer.model(data)

            # Attach meta information
            ref_data = data[0] if isinstance(data, typing.Sequence) else data
            for key in ['im_shape', 'scale_factor', 'im_id', 'bbox_num']:
                if key in ref_data:
                    outs[key] = ref_data[key]

            outs = ensure_numpy(outs)
            if 'bbox_num' not in outs:
                logger.warning("Model output missing 'bbox_num'. Skip this batch.")
                continue

            bbox_num = np.array(outs['bbox_num']).astype(np.int32).flatten()
            outs['bbox_num'] = bbox_num
            if bbox_num.sum() == 0 and args.skip_empty:
                continue

            batch_res = get_infer_results(outs, clsid2catid)
            im_ids = outs['im_id']
            if im_ids.ndim > 1:
                im_ids = im_ids.flatten()

            start = 0
            for idx, im_id in enumerate(im_ids):
                im_id_int = int(im_id)
                radar_path = imid2path.get(im_id_int, None)
                if radar_path is None or not os.path.exists(radar_path):
                    logger.warning(f"Missing radar image for im_id={im_id_int}, skip.")
                    start += bbox_num[idx]
                    continue

                try:
                    image = Image.open(radar_path).convert('RGB')
                except Exception as exc:
                    logger.warning(f"Failed to open {radar_path}: {exc}")
                    start += bbox_num[idx]
                    continue

                end = start + int(bbox_num[idx])
                bbox_res = batch_res['bbox'][start:end] if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] if 'segm' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] if 'keypoint' in batch_res else None
                pose3d_res = batch_res['pose3d'][start:end] if 'pose3d' in batch_res else None

                image = visualize_results(
                    image,
                    bbox_res,
                    mask_res,
                    segm_res,
                    keypoint_res,
                    pose3d_res,
                    im_id_int,
                    catid2name,
                    threshold=args.draw_threshold)

                save_path = os.path.join(args.output_dir, f"{im_id_int:06d}.png")
                image.save(save_path, format='PNG')
                logger.info(f"Saved visualization: {save_path}")

                saved_images += 1
                start = end
                if args.max_images > 0 and saved_images >= args.max_images:
                    return


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
            "Model weights must be specified via the config file or --weights.")

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)

    visualize_predictions(cfg, args)


if __name__ == "__main__":
    main()
