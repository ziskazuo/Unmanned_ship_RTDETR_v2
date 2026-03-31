#!/usr/bin/env python3
"""Class-agnostic ship-only evaluation for radar-camera RT-DETR."""

import argparse
import json
import os
import sys
import typing

import numpy as np
import paddle

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.utils.logger import setup_logger
from ppdet.utils.check import check_config, check_gpu, check_npu, check_xpu, check_mlu
from ppdet.metrics.coco_utils import get_infer_results
from ppdet.metrics.map_utils import DetectionMAP, prune_zero_padding

logger = setup_logger('eval_ship_only')


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--output_json', type=str, default=None)
    return parser.parse_args()


def ensure_numpy(data):
    if isinstance(data, paddle.Tensor):
        return data.numpy()
    if isinstance(data, typing.Sequence) and not isinstance(data, (str, bytes)):
        return [ensure_numpy(d) for d in data]
    if isinstance(data, dict):
        return {k: ensure_numpy(v) for k, v in data.items()}
    return data


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
        paddle.set_device('gpu:0')
    elif cfg.use_npu:
        paddle.set_device('npu')
    elif cfg.use_xpu:
        paddle.set_device('xpu')
    elif cfg.use_mlu:
        paddle.set_device('mlu')
    else:
        paddle.set_device('cpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)

    init_parallel_env()
    trainer = Trainer(cfg, mode='eval')
    trainer.load_weights(cfg.weights)
    trainer.model.eval()

    detection_map = DetectionMAP(
        class_num=1,
        overlap_thresh=0.5,
        map_type='11point',
        is_bbox_normalized=False,
        evaluate_difficult=False,
        catid2name={1: 'ship'},
        classwise=False)

    dataset = trainer.dataset
    loader = trainer.loader
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}

    with paddle.no_grad():
        for step_id, data in enumerate(loader):
            outputs = trainer.model(data)
            batch = data[0] if isinstance(data, typing.Sequence) else data
            outs = ensure_numpy(outputs)
            im_id = batch['im_id']
            outs['im_id'] = ensure_numpy(im_id)
            infer_results = get_infer_results(outs, clsid2catid)
            infer_results = infer_results['bbox'] if 'bbox' in infer_results else []

            gt_boxes = batch['gt_poly']
            gt_labels = batch['gt_class']
            if 'scale_factor' in batch:
                scale_factor = ensure_numpy(batch['scale_factor'])
            else:
                scale_factor = np.ones((len(gt_boxes), 2), dtype='float32')

            for i in range(len(gt_boxes)):
                gt_box = ensure_numpy(gt_boxes[i])
                h, w = scale_factor[i]
                gt_box = gt_box / np.array([w, h, w, h, w, h, w, h], dtype='float32')
                gt_label = ensure_numpy(gt_labels[i])
                gt_box, gt_label, _ = prune_zero_padding(gt_box, gt_label)
                gt_label = np.zeros_like(gt_label, dtype=np.int32)

                image_id = int(outs['im_id'][i]) if np.ndim(outs['im_id']) == 1 else int(outs['im_id'][i][0])
                bbox = [res['bbox'] for res in infer_results if int(res['image_id']) == image_id]
                score = [res['score'] for res in infer_results if int(res['image_id']) == image_id]
                label = [0 for res in infer_results if int(res['image_id']) == image_id]
                detection_map.update(bbox, score, label, gt_box, gt_label)

            if step_id % 20 == 0:
                logger.info('Ship-only eval iter: {}'.format(step_id))

    detection_map.accumulate()
    result = {
        'ship_mAP_50_11point': round(float(detection_map.get_map() * 100.0), 2),
        'overlap_thresh': 0.5,
        'map_type': '11point',
    }
    logger.info('ship_mAP(0.50, 11point) = {:.2f}%'.format(result['ship_mAP_50_11point']))
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
