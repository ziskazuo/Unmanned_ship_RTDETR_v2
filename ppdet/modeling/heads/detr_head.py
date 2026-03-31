# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ppdet.core.workspace import register
import pycocotools.mask as mask_util
from ..initializer import linear_init_, constant_
from ..transformers.utils import inverse_sigmoid

__all__ = ['DETRHead', 'DeformableDETRHead', 'DINOHead', 'MaskDINOHead',
           'DINOHead_Rotate', 'DINOHead_Rotate_RouteROI']


class MLP(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiHeadAttentionMap(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0,
                 bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant()) if bias else False

        self.q_proj = nn.Linear(query_dim, hidden_dim, weight_attr, bias_attr)
        self.k_proj = nn.Conv2D(
            query_dim,
            hidden_dim,
            1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        bs, num_queries, n, c, h, w = q.shape[0], q.shape[1], self.num_heads,\
                                      self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]
        qh = q.reshape([bs, num_queries, n, c])
        kh = k.reshape([bs, n, c, h, w])
        # weights = paddle.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        qh = qh.transpose([0, 2, 1, 3]).reshape([-1, num_queries, c])
        kh = kh.reshape([-1, c, h * w])
        weights = paddle.bmm(qh * self.normalize_fact, kh).reshape(
            [bs, n, num_queries, h, w]).transpose([0, 2, 1, 3, 4])

        if mask is not None:
            weights += mask
        # fix a potenial bug: https://github.com/facebookresearch/detr/issues/247
        weights = F.softmax(weights.flatten(3), axis=-1).reshape(weights.shape)
        weights = self.dropout(weights)
        return weights


class MaskHeadFPNConv(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/segmentation.py

        Simple convolutional head, using group norm.
        Upsampling is done using a FPN approach
    """

    def __init__(self, input_dim, fpn_dims, context_dim, num_groups=8):
        super().__init__()

        inter_dims = [input_dim,
                      ] + [context_dim // (2**i) for i in range(1, 5)]
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Constant())

        self.conv0 = self._make_layers(input_dim, input_dim, 3, num_groups,
                                       weight_attr, bias_attr)
        self.conv_inter = nn.LayerList()
        for in_dims, out_dims in zip(inter_dims[:-1], inter_dims[1:]):
            self.conv_inter.append(
                self._make_layers(in_dims, out_dims, 3, num_groups, weight_attr,
                                  bias_attr))

        self.conv_out = nn.Conv2D(
            inter_dims[-1],
            1,
            3,
            padding=1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.adapter = nn.LayerList()
        for i in range(len(fpn_dims)):
            self.adapter.append(
                nn.Conv2D(
                    fpn_dims[i],
                    inter_dims[i + 1],
                    1,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr))

    def _make_layers(self,
                     in_dims,
                     out_dims,
                     kernel_size,
                     num_groups,
                     weight_attr=None,
                     bias_attr=None):
        return nn.Sequential(
            nn.Conv2D(
                in_dims,
                out_dims,
                kernel_size,
                padding=kernel_size // 2,
                weight_attr=weight_attr,
                bias_attr=bias_attr),
            nn.GroupNorm(num_groups, out_dims),
            nn.ReLU())

    def forward(self, x, bbox_attention_map, fpns):
        x = paddle.concat([
            x.tile([bbox_attention_map.shape[1], 1, 1, 1]),
            bbox_attention_map.flatten(0, 1)
        ], 1)
        x = self.conv0(x)
        for inter_layer, adapter_layer, feat in zip(self.conv_inter[:-1],
                                                    self.adapter, fpns):
            feat = adapter_layer(feat).tile(
                [bbox_attention_map.shape[1], 1, 1, 1])
            x = inter_layer(x)
            x = feat + F.interpolate(x, size=feat.shape[-2:])

        x = self.conv_inter[-1](x)
        x = self.conv_out(x)
        return x


@register
class DETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'use_focal_loss']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 nhead=8,
                 num_mlp_layers=3,
                 loss='DETRLoss',
                 fpn_dims=[1024, 512, 256],
                 with_mask_head=False,
                 use_focal_loss=False):
        super(DETRHead, self).__init__()
        # add background class
        self.num_classes = num_classes if use_focal_loss else num_classes + 1
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.with_mask_head = with_mask_head
        self.use_focal_loss = use_focal_loss

        self.score_head = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)
        if self.with_mask_head:
            self.bbox_attention = MultiHeadAttentionMap(hidden_dim, hidden_dim,
                                                        nhead)
            self.mask_head = MaskHeadFPNConv(hidden_dim + nhead, fpn_dims,
                                             hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.score_head)

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):

        return {
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'fpn_dims': [i.channels for i in input_shape[::-1]][1:]
        }

    @staticmethod
    def get_gt_mask_from_polygons(gt_poly, pad_mask):
        out_gt_mask = []
        for polygons, padding in zip(gt_poly, pad_mask):
            height, width = int(padding[:, 0].sum()), int(padding[0, :].sum())
            masks = []
            for obj_poly in polygons:
                rles = mask_util.frPyObjects(obj_poly, height, width)
                rle = mask_util.merge(rles)
                masks.append(
                    paddle.to_tensor(mask_util.decode(rle)).astype('float32'))
            masks = paddle.stack(masks)
            masks_pad = paddle.zeros(
                [masks.shape[0], pad_mask.shape[1], pad_mask.shape[2]])
            masks_pad[:, :height, :width] = masks
            out_gt_mask.append(masks_pad)
        return out_gt_mask

    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size, hidden_dim, h, w],
                            src_proj: [batch_size, h*w, hidden_dim],
                            src_mask: [batch_size, 1, 1, h, w])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, src_proj, src_mask = out_transformer
        outputs_logit = self.score_head(feats)
        outputs_bbox = F.sigmoid(self.bbox_head(feats))
        outputs_seg = None
        if self.with_mask_head:
            bbox_attention_map = self.bbox_attention(feats[-1], memory,
                                                     src_mask)
            fpn_feats = [a for a in body_feats[::-1]][1:]
            outputs_seg = self.mask_head(src_proj, bbox_attention_map,
                                         fpn_feats)
            outputs_seg = outputs_seg.reshape([
                feats.shape[1], feats.shape[2], outputs_seg.shape[-2],
                outputs_seg.shape[-1]
            ])

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            gt_mask = self.get_gt_mask_from_polygons(
                inputs['gt_poly'],
                inputs['pad_mask']) if 'gt_poly' in inputs else None
            return self.loss(
                outputs_bbox,
                outputs_logit,
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=outputs_seg,
                gt_mask=gt_mask)
        else:
            return (outputs_bbox[-1], outputs_logit[-1], outputs_seg)


@register
class DeformableDETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=512,
                 nhead=8,
                 num_mlp_layers=3,
                 loss='DETRLoss'):
        super(DeformableDETRHead, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.loss = loss

        self.score_head = nn.Linear(hidden_dim, self.num_classes)
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.score_head)
        constant_(self.score_head.bias, -4.595)
        constant_(self.bbox_head.layers[-1].weight)

        with paddle.no_grad():
            bias = paddle.zeros_like(self.bbox_head.layers[-1].bias)
            bias[2:] = -2.0
            self.bbox_head.layers[-1].bias.set_value(bias)

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):
        return {'hidden_dim': hidden_dim, 'nhead': nhead}

    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size,
                                \sum_{l=0}^{L-1} H_l \cdot W_l, hidden_dim],
                            reference_points: [batch_size, num_queries, 2])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, reference_points = out_transformer
        reference_points = inverse_sigmoid(reference_points.unsqueeze(0))
        outputs_bbox = self.bbox_head(feats)

        # It's equivalent to "outputs_bbox[:, :, :, :2] += reference_points",
        # but the gradient is wrong in paddle.
        outputs_bbox = paddle.concat(
            [
                outputs_bbox[:, :, :, :2] + reference_points,
                outputs_bbox[:, :, :, 2:]
            ],
            axis=-1)

        outputs_bbox = F.sigmoid(outputs_bbox)
        outputs_logit = self.score_head(feats)

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            return self.loss(outputs_bbox, outputs_logit, inputs['gt_bbox'],
                             inputs['gt_class'])
        else:
            return (outputs_bbox[-1], outputs_logit[-1], None)


@register
class DINOHead_Rotate(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss_Rotate'):
        super(DINOHead_Rotate, self).__init__()
        self.loss = loss


    def forward(self, out_transformer, body_feats, inputs=None, flag=None):


        (dec_out_bboxes, dec_out_logits, dec_out_angles_cls, dec_out_angles, enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls,
         angle_max,angle_proj,
         dn_meta) = out_transformer

        b, l = enc_topk_angles_cls.shape[:2]
        enc_topk_angles = F.softmax(enc_topk_angles_cls.reshape([b, l, 1, angle_max + 1
                                                 ])).matmul(angle_proj)
        dec_out_bboxes = paddle.concat([dec_out_bboxes,dec_out_angles], axis=-1)
        enc_out_bboxes = paddle.concat([enc_topk_bboxes, enc_topk_angles], axis=-1)

        if self.training:
            assert inputs is not None
            assert 'gt_rbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                if isinstance(dn_meta, list):
                    dual_groups = len(dn_meta) - 1
                    dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dual_groups + 1, axis=2)
                    dec_out_logits = paddle.split(
                        dec_out_logits, dual_groups + 1, axis=2)
                    enc_topk_bboxes = paddle.split(
                        enc_topk_bboxes, dual_groups + 1, axis=1)
                    enc_topk_logits = paddle.split(
                        enc_topk_logits, dual_groups + 1, axis=1)

                    dec_out_bboxes_list = []
                    dec_out_logits_list = []
                    dn_out_bboxes_list = []
                    dn_out_logits_list = []
                    loss = {}
                    for g_id in range(dual_groups + 1):
                        if dn_meta[g_id] is not None:
                            dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(
                                dec_out_bboxes[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_logits_gid, dec_out_logits_gid = paddle.split(
                                dec_out_logits[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                        else:
                            dn_out_bboxes_gid, dn_out_logits_gid = None, None
                            dec_out_bboxes_gid = dec_out_bboxes[g_id]
                            dec_out_logits_gid = dec_out_logits[g_id]
                        out_bboxes_gid = paddle.concat([
                            enc_topk_bboxes[g_id].unsqueeze(0),
                            dec_out_bboxes_gid
                        ])
                        out_logits_gid = paddle.concat([
                            enc_topk_logits[g_id].unsqueeze(0),
                            dec_out_logits_gid
                        ])
                        loss_gid = self.loss(
                            out_bboxes_gid,
                            out_logits_gid,
                            inputs['gt_rbox'],
                            inputs['gt_class'],
                            dn_out_bboxes=dn_out_bboxes_gid,
                            dn_out_logits=dn_out_logits_gid,
                            dn_meta=dn_meta[g_id])
                        # sum loss
                        for key, value in loss_gid.items():
                            loss.update({
                                key: loss.get(key, paddle.zeros([1])) + value
                            })

                    # average across (dual_groups + 1)
                    for key, value in loss.items():
                        loss.update({key: value / (dual_groups + 1)})
                    return loss
                else:
                    dn_out_bboxes, dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                    dn_out_logits, dec_out_logits = paddle.split(
                        dec_out_logits, dn_meta['dn_num_split'], axis=2)
                    dn_out_angles_cls, dec_out_angles_cls = paddle.split(
                        dec_out_angles_cls, dn_meta['dn_num_split'], axis=2)


            else:
                dn_out_bboxes, dn_out_logits, dn_out_angles_cls, dn_out_angles = None, None, None, None

            out_bboxes = paddle.concat(
                [enc_out_bboxes.unsqueeze(0), dec_out_bboxes])
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits])
            out_angles_cls = paddle.concat(
                [enc_topk_angles_cls.unsqueeze(0), dec_out_angles_cls]
            )
            # `im_shape` may come in as [h, w] or [b, 2] depending on reader/collate.
            im_shape = paddle.to_tensor(inputs['im_shape'])
            if len(im_shape.shape) > 1:
                im_shape = im_shape[0]
            im_shape = paddle.stack([im_shape[1], im_shape[0]])

            return self.loss(
                out_bboxes,
                out_logits,
                out_angles_cls,
                inputs['gt_rbox'],
                inputs['gt_class'],
                im_shape,
                dn_out_bboxes=dn_out_bboxes,
                dn_out_logits=dn_out_logits,
                dn_out_angle_cls=dn_out_angles_cls,
                dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], None)


@register
class DINOHead_Rotate_RouteROI(nn.Layer):
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=4,
                 hidden_dim=256,
                 camera_feat_strides=[8, 16, 32],
                 roi_resolution=7,
                 route_invalid_logit_bias=2.0,
                 visible_logit_weight=1.0,
                 loss='DINOLoss_Rotate_RouteROI'):
        super(DINOHead_Rotate_RouteROI, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.camera_feat_strides = camera_feat_strides
        self.roi_resolution = roi_resolution
        self.route_invalid_logit_bias = float(route_invalid_logit_bias)
        self.visible_logit_weight = float(visible_logit_weight)
        self.loss = loss

        route_in_dim = hidden_dim + 5 + 4 + 16
        self.route_trunk = MLP(route_in_dim, hidden_dim, hidden_dim, num_layers=2)
        self.route_primary_head = nn.Linear(hidden_dim, 5)
        self.visible_head = nn.Linear(hidden_dim, 4)
        self.coarse_box_embed = MLP(hidden_dim + hidden_dim * 3 + 4, hidden_dim, 4, num_layers=2)
        self.cam_cls_head = nn.Linear(hidden_dim * 3, num_classes)
        self.fuse_cls_head = MLP(hidden_dim + hidden_dim * 3 + 5, hidden_dim, num_classes, num_layers=2)
        self._reset_parameters()

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):
        return {
            'hidden_dim': hidden_dim,
            'camera_feat_strides': [i.stride for i in input_shape]
        }

    def _reset_parameters(self):
        linear_init_(self.route_primary_head)
        linear_init_(self.visible_head)
        linear_init_(self.cam_cls_head)

    def _ensure_batch_tensor(self, value, batch_size=None, dtype='float32'):
        tensor = paddle.to_tensor(value, dtype=dtype)
        if batch_size is not None and tensor.shape[0] != batch_size:
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0).tile([batch_size, 1])
            elif len(tensor.shape) == 2 and tensor.shape[0] == 1:
                tensor = tensor.tile([batch_size, 1])
        return tensor

    def _radar_hw(self, inputs, batch_size):
        im_shape = self._ensure_batch_tensor(inputs['im_shape'], batch_size=batch_size)
        if len(im_shape.shape) == 1:
            im_shape = im_shape.unsqueeze(0).tile([batch_size, 1])
        if im_shape.shape[-1] != 2:
            raise ValueError('im_shape must contain [h, w].')
        radar_hw = paddle.stack([im_shape[:, 1], im_shape[:, 0]], axis=-1)
        return radar_hw

    def _radar_boxes_to_pixel_rbox(self, boxes, angles, radar_hw):
        radar_hw = radar_hw.unsqueeze(1)
        cx = boxes[..., 0] * radar_hw[..., 0]
        cy = boxes[..., 1] * radar_hw[..., 1]
        w = boxes[..., 2] * radar_hw[..., 0]
        h = boxes[..., 3] * radar_hw[..., 1]
        angle = angles.squeeze(-1)
        return paddle.stack([cx, cy, w, h, angle], axis=-1)

    def _rbox_to_corners(self, rbox):
        cx, cy, w, h, angle = [rbox[..., i] for i in range(5)]
        half_w = w / 2.0
        half_h = h / 2.0
        base = paddle.to_tensor(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
            dtype=rbox.dtype)
        base = base.reshape([1, 1, 4, 2])
        offsets = paddle.stack([half_w, half_h], axis=-1).unsqueeze(-2) * base
        cos_a = paddle.cos(angle).unsqueeze(-1)
        sin_a = paddle.sin(angle).unsqueeze(-1)
        x = offsets[..., 0]
        y = offsets[..., 1]
        rot_x = x * cos_a - y * sin_a
        rot_y = x * sin_a + y * cos_a
        center = paddle.stack([cx, cy], axis=-1).unsqueeze(-2)
        return paddle.stack([rot_x, rot_y], axis=-1) + center

    def _radar_pixels_to_metric(self, corners_uv, radar_resolution, lidar_range):
        cx = radar_resolution / 2.0
        cy = radar_resolution / 2.0
        scale = (radar_resolution / 2.0) / paddle.clip(lidar_range, min=1e-6)
        x = (cy.unsqueeze(-1).unsqueeze(-1) - corners_uv[..., 1]) / scale.unsqueeze(-1).unsqueeze(-1)
        y = (corners_uv[..., 0] - cx.unsqueeze(-1).unsqueeze(-1)) / scale.unsqueeze(-1).unsqueeze(-1)
        return paddle.stack([x, y], axis=-1)

    def _project_metric_corners(self, metric_xy, inputs):
        # Some training branches still carry a singleton decoder dimension here.
        # RouteROI projection expects [B, Q, 4, 2], so collapse [B, 1, Q, 4, 2].
        if len(metric_xy.shape) == 5 and metric_xy.shape[1] == 1:
            metric_xy = metric_xy.squeeze(1)
        batch_size = metric_xy.shape[0]
        num_queries = metric_xy.shape[1]
        num_cameras = 4
        camera_intrinsics = self._ensure_batch_tensor(
            inputs['camera_intrinsics'], batch_size=batch_size)
        camera_extrinsics = self._ensure_batch_tensor(
            inputs['camera_extrinsics'], batch_size=batch_size)
        camera_img_size = self._ensure_batch_tensor(
            inputs['camera_img_size'], batch_size=batch_size)
        projection_plane_height = paddle.to_tensor(
            inputs.get('projection_plane_height', -6.0), dtype='float32')
        if len(projection_plane_height.shape) == 0:
            projection_plane_height = projection_plane_height.reshape([1]).tile([batch_size])
        elif len(projection_plane_height.shape) > 1:
            projection_plane_height = projection_plane_height.reshape([batch_size, -1])[:, 0]

        z = projection_plane_height.reshape([batch_size, 1, 1, 1]).tile([1, num_queries, 4, 1])
        ones = paddle.ones_like(z)
        points = paddle.concat([metric_xy, z, ones], axis=-1)
        points = points.unsqueeze(2).unsqueeze(-1).tile([1, 1, num_cameras, 1, 1, 1])
        ext = camera_extrinsics.unsqueeze(1).unsqueeze(3)
        cam_points = paddle.matmul(ext, points).squeeze(-1)
        cam_x = cam_points[..., 0]
        cam_y = cam_points[..., 1]
        cam_z = cam_points[..., 2]

        fx = camera_intrinsics[:, :, 0, 0].unsqueeze(1).unsqueeze(-1)
        fy = camera_intrinsics[:, :, 1, 1].unsqueeze(1).unsqueeze(-1)
        cx = camera_intrinsics[:, :, 0, 2].unsqueeze(1).unsqueeze(-1)
        cy = camera_intrinsics[:, :, 1, 2].unsqueeze(1).unsqueeze(-1)

        safe_z = paddle.where(cam_z.abs() < 1e-6, paddle.full_like(cam_z, 1e-6), cam_z)
        u = fx * cam_x / safe_z + cx
        v = fy * cam_y / safe_z + cy
        corners_2d = paddle.stack([u, v], axis=-1)

        valid_depth = cam_z > 1e-4
        width = camera_img_size[:, :, 0].unsqueeze(1)
        height = camera_img_size[:, :, 1].unsqueeze(1)

        xmin = corners_2d[..., 0].min(axis=-1)
        xmax = corners_2d[..., 0].max(axis=-1)
        ymin = corners_2d[..., 1].min(axis=-1)
        ymax = corners_2d[..., 1].max(axis=-1)
        xmin = paddle.clip(xmin, min=0.0)
        ymin = paddle.clip(ymin, min=0.0)
        xmax = paddle.minimum(xmax, width - 1.0)
        ymax = paddle.minimum(ymax, height - 1.0)
        boxes = paddle.stack([xmin, ymin, xmax, ymax], axis=-1)

        visible = valid_depth.all(axis=-1)
        visible = paddle.logical_and(visible, xmax > xmin)
        visible = paddle.logical_and(visible, ymax > ymin)
        return boxes, corners_2d, visible.astype('float32')

    def _roi_align_single_level(self, feat, boxes, batch_ids, spatial_scale):
        if boxes.shape[0] == 0:
            return paddle.zeros(
                [0, feat.shape[1], self.roi_resolution, self.roi_resolution],
                dtype=feat.dtype)

        batch_ids_np = np.asarray(batch_ids.numpy(), dtype=np.int64)
        order = np.argsort(batch_ids_np)
        restore = np.argsort(order)
        boxes_sorted = paddle.gather(boxes, paddle.to_tensor(order, dtype='int64'))
        counts = [0] * int(feat.shape[0])
        for batch_id in batch_ids_np[order]:
            counts[int(batch_id)] += 1
        boxes_num = paddle.to_tensor(counts, dtype='int32')
        roi_feat = paddle.vision.ops.roi_align(
            x=feat,
            boxes=boxes_sorted,
            boxes_num=boxes_num,
            output_size=self.roi_resolution,
            spatial_scale=spatial_scale,
            aligned=False)
        return paddle.gather(roi_feat, paddle.to_tensor(restore, dtype='int64'))

    def _multi_level_roi_feature(self, camera_feats, boxes, batch_ids):
        if boxes.shape[0] == 0:
            channels = camera_feats[0].shape[1] * len(camera_feats)
            return paddle.zeros([0, channels], dtype=camera_feats[0].dtype)

        pooled = []
        for feat, stride in zip(camera_feats, self.camera_feat_strides):
            roi_feat = self._roi_align_single_level(
                feat, boxes, batch_ids, spatial_scale=1.0 / float(stride))
            pooled.append(F.adaptive_avg_pool2d(roi_feat, output_size=1).flatten(1))
        return paddle.concat(pooled, axis=-1)

    def _extract_camera_roi_features(self,
                                     camera_feats,
                                     query_feat,
                                     geom_boxes,
                                     camera_img_size):
        batch_size, num_queries, num_cameras, _ = geom_boxes.shape
        feat_dim = camera_feats[0].shape[1] * len(camera_feats)
        per_camera_roi_feat = paddle.zeros(
            [batch_size, num_queries, num_cameras, feat_dim], dtype=query_feat.dtype)
        refined_boxes = paddle.zeros_like(geom_boxes)
        for camera_id in range(num_cameras):
            boxes = geom_boxes[:, :, camera_id, :].reshape([-1, 4])
            valid = paddle.logical_and(boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1])
            valid_idx = paddle.nonzero(valid).flatten()
            if valid_idx.shape[0] == 0:
                continue

            boxes_valid = paddle.gather(boxes, valid_idx)
            query_valid = paddle.gather(query_feat.reshape([-1, query_feat.shape[-1]]), valid_idx)
            batch_index = (valid_idx // num_queries).astype('int64')
            batch_ids = batch_index * 4 + camera_id
            roi_feat = self._multi_level_roi_feature(camera_feats, boxes_valid, batch_ids)

            wh = paddle.gather(
                camera_img_size[:, camera_id, :], batch_index, axis=0)
            whxy = paddle.concat([wh[:, 0:1], wh[:, 1:2], wh[:, 0:1], wh[:, 1:2]], axis=-1)
            coarse_norm = boxes_valid / paddle.clip(whxy, min=1e-6)
            delta = 0.1 * paddle.tanh(
                self.coarse_box_embed(paddle.concat([query_valid, roi_feat, coarse_norm], axis=-1)))
            refined_norm = paddle.clip(coarse_norm + delta, min=0.0, max=1.0)
            refined_valid = refined_norm * whxy
            refined_roi_feat = self._multi_level_roi_feature(
                camera_feats, refined_valid, batch_ids)

            feat_index = paddle.concat([valid_idx.unsqueeze(-1), paddle.full([valid_idx.shape[0], 1], camera_id, dtype='int64')], axis=-1)
            per_camera_roi_feat = paddle.scatter_nd_add(
                per_camera_roi_feat.reshape([-1, num_cameras, feat_dim]),
                feat_index,
                refined_roi_feat)
            refined_boxes = paddle.scatter_nd_add(
                refined_boxes.reshape([-1, num_cameras, 4]),
                feat_index,
                refined_valid)

        per_camera_roi_feat = per_camera_roi_feat.reshape([batch_size, num_queries, num_cameras, feat_dim])
        refined_boxes = refined_boxes.reshape([batch_size, num_queries, num_cameras, 4])
        return per_camera_roi_feat, refined_boxes

    def forward(self, out_transformer, body_feats, inputs=None, flag=None):
        (dec_out_bboxes, dec_out_logits, dec_out_angles_cls, dec_out_angles,
         enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls, angle_max,
         angle_proj, dn_meta, last_query_feat) = out_transformer

        dn_out_bboxes = None
        dn_out_logits = None
        dn_out_angles_cls = None
        if self.training and dn_meta is not None and not isinstance(dn_meta, list):
            dn_out_bboxes, dec_out_bboxes = paddle.split(
                dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
            dn_out_logits, dec_out_logits = paddle.split(
                dec_out_logits, dn_meta['dn_num_split'], axis=2)
            dn_out_angles_cls, dec_out_angles_cls = paddle.split(
                dec_out_angles_cls, dn_meta['dn_num_split'], axis=2)
            _, dec_out_angles = paddle.split(
                dec_out_angles, dn_meta['dn_num_split'], axis=2)
            _, last_query_feat = paddle.split(
                last_query_feat, dn_meta['dn_num_split'], axis=1)

        radar_logits = dec_out_logits[-1]
        radar_angles = dec_out_angles[-1]
        radar_boxes = dec_out_bboxes[-1]
        batch_size, num_queries = radar_boxes.shape[:2]
        radar_hw = self._radar_hw(inputs, batch_size)
        radar_rbox_pixels = self._radar_boxes_to_pixel_rbox(
            radar_boxes, radar_angles, radar_hw)

        radar_resolution = paddle.full(
            [batch_size], float(radar_hw[0, 0]), dtype='float32')
        lidar_range = paddle.to_tensor(inputs.get('lidar_range', 4000.0), dtype='float32')
        if len(lidar_range.shape) == 0:
            lidar_range = lidar_range.reshape([1]).tile([batch_size])
        elif len(lidar_range.shape) > 1:
            lidar_range = lidar_range.reshape([batch_size, -1])[:, 0]
        radar_corners = self._rbox_to_corners(radar_rbox_pixels)
        metric_corners = self._radar_pixels_to_metric(
            radar_corners, radar_resolution, lidar_range)
        geom_boxes, _geom_corners, geom_visible = self._project_metric_corners(
            metric_corners, inputs)

        camera_img_size = self._ensure_batch_tensor(
            inputs['camera_img_size'], batch_size=batch_size)
        camera_wh = paddle.stack(
            [camera_img_size[:, :, 0], camera_img_size[:, :, 1],
             camera_img_size[:, :, 0], camera_img_size[:, :, 1]], axis=-1)
        geom_boxes_norm = geom_boxes / paddle.clip(camera_wh.unsqueeze(1), min=1e-6)
        route_input = paddle.concat([
            last_query_feat,
            paddle.concat([radar_boxes, radar_angles], axis=-1),
            geom_visible,
            geom_boxes_norm.reshape([batch_size, num_queries, -1]),
        ], axis=-1)
        route_feat = self.route_trunk(route_input)
        visible_logits = self.visible_head(route_feat)
        route_primary_logits = self.route_primary_head(route_feat)

        # Geometry acts as a soft prior, not a hard veto. We keep the raw
        # route logits for CE supervision and only apply a mild bias to the
        # inference/routing branch so slightly mismatched calibration does not
        # make a ground-truth route impossible to learn.
        route_camera_logits = route_primary_logits[:, :, :4] + \
            self.visible_logit_weight * visible_logits - \
            (1.0 - geom_visible) * self.route_invalid_logit_bias
        route_logits_soft = paddle.concat(
            [route_camera_logits, route_primary_logits[:, :, 4:5]], axis=-1)
        route_prob = F.softmax(route_logits_soft, axis=-1)

        per_camera_roi_feat, refined_boxes = self._extract_camera_roi_features(
            body_feats, last_query_feat, geom_boxes, camera_img_size)
        per_camera_roi_feat = per_camera_roi_feat * geom_visible.unsqueeze(-1)
        refined_boxes = refined_boxes * geom_visible.unsqueeze(-1)
        per_camera_cls_logits = self.cam_cls_head(per_camera_roi_feat)

        selected_camera = paddle.argmax(route_logits_soft, axis=-1)
        selected_camera_clamped = paddle.clip(selected_camera, min=0, max=3)
        gather_idx = selected_camera_clamped.unsqueeze(-1).unsqueeze(-1).tile(
            [1, 1, 1, per_camera_roi_feat.shape[-1]])
        selected_camera_feat = paddle.take_along_axis(
            per_camera_roi_feat, gather_idx, axis=2).squeeze(2)
        selected_visible = paddle.take_along_axis(
            geom_visible,
            selected_camera_clamped.unsqueeze(-1),
            axis=2).squeeze(-1)

        fuse_input = paddle.concat([last_query_feat, selected_camera_feat, route_prob], axis=-1)
        fused_delta = self.fuse_cls_head(fuse_input)
        use_camera = paddle.logical_and(
            selected_camera != 4, selected_visible > 0.5).astype(
                fused_delta.dtype).unsqueeze(-1)
        fused_logits = radar_logits + fused_delta * use_camera

        if self.training:
            assert inputs is not None
            out_bboxes = paddle.concat(
                [paddle.concat([enc_topk_bboxes, F.softmax(enc_topk_angles_cls.reshape([
                    batch_size, enc_topk_angles_cls.shape[1], 1, angle_max + 1]), axis=-1).matmul(angle_proj)], axis=-1).unsqueeze(0),
                 paddle.concat([dec_out_bboxes, dec_out_angles], axis=-1)],
                axis=0)
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits[:-1], fused_logits.unsqueeze(0)],
                axis=0)
            out_angles_cls = paddle.concat(
                [enc_topk_angles_cls.unsqueeze(0), dec_out_angles_cls], axis=0)

            im_shape = paddle.to_tensor(inputs['im_shape'], dtype='float32')
            if len(im_shape.shape) == 1:
                im_shape = im_shape.unsqueeze(0)
            if im_shape.shape[0] == 1 and batch_size > 1:
                im_shape = im_shape.tile([batch_size, 1])
            ref_im_shape = im_shape[0:1]
            if not paddle.allclose(im_shape, ref_im_shape.tile([im_shape.shape[0], 1])):
                raise ValueError(
                    "RouteROI training currently expects a fixed radar image "
                    "shape across the whole batch.")
            matcher_im_shape = paddle.stack([ref_im_shape[0, 1], ref_im_shape[0, 0]])

            return self.loss(
                out_bboxes,
                out_logits,
                out_angles_cls,
                inputs['gt_rbox'],
                inputs['gt_class'],
                matcher_im_shape,
                dn_meta=dn_meta,
                dn_out_bboxes=dn_out_bboxes,
                dn_out_logits=dn_out_logits,
                dn_out_angle_cls=dn_out_angles_cls,
                route_primary_logits=route_primary_logits,
                visible_logits=visible_logits,
                camera_cls_logits=per_camera_cls_logits,
                camera_refined_boxes=refined_boxes,
                inputs=inputs)
        final_boxes = paddle.concat([radar_boxes, radar_angles], axis=-1)
        return (final_boxes, fused_logits, None)
@register
class DINOHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss',group=False,groupx4=False,groupx5=False,groupx3_v3=False,groupx3_v4=False,groupx3_v5=False,
                 groupx3_v6=False):
        super(DINOHead, self).__init__()
        self.loss = loss
        self.group = group
        self.groupx4 = groupx4
        self.groupx5 = groupx5
        self.groupx3_v3 = groupx3_v3
        self.groupx3_v4 = groupx3_v4
        self.groupx3_v5 = groupx3_v5
        self.groupx3_v6 = groupx3_v6

    def forward(self, out_transformer, body_feats, inputs=None):
        if self.groupx5:
            (dec_out_bboxes, dec_out_logits, dec_out, enc_topk_bboxes, enc_topk_logits,
             dn_meta) = out_transformer
        elif self.groupx3_v3:
            (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
             dn_meta,all_feats) = out_transformer
        elif self.groupx3_v4:
            (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
             dn_meta, all_feats, generate_vis_pic,generate_ir_pic) = out_transformer
        elif self.groupx3_v5:
            (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
             dn_meta, all_feats, generate_vis_pic,generate_ir_pic) = out_transformer
        elif self.groupx3_v6:
            (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
             dn_meta, all_feats, generate_vis_pic,generate_ir_pic) = out_transformer
        else:
            (dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits,
             dn_meta) = out_transformer
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs

            if dn_meta is not None:
                if isinstance(dn_meta, list):
                    dual_groups = len(dn_meta) - 1
                    dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dual_groups + 1, axis=2)
                    dec_out_logits = paddle.split(
                        dec_out_logits, dual_groups + 1, axis=2)
                    enc_topk_bboxes = paddle.split(
                        enc_topk_bboxes, dual_groups + 1, axis=1)
                    enc_topk_logits = paddle.split(
                        enc_topk_logits, dual_groups + 1, axis=1)

                    dec_out_bboxes_list = []
                    dec_out_logits_list = []
                    dn_out_bboxes_list = []
                    dn_out_logits_list = []
                    loss = {}
                    for g_id in range(dual_groups + 1):
                        if dn_meta[g_id] is not None:
                            dn_out_bboxes_gid, dec_out_bboxes_gid = paddle.split(
                                dec_out_bboxes[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                            dn_out_logits_gid, dec_out_logits_gid = paddle.split(
                                dec_out_logits[g_id],
                                dn_meta[g_id]['dn_num_split'],
                                axis=2)
                        else:
                            dn_out_bboxes_gid, dn_out_logits_gid = None, None
                            dec_out_bboxes_gid = dec_out_bboxes[g_id]
                            dec_out_logits_gid = dec_out_logits[g_id]
                        out_bboxes_gid = paddle.concat([
                            enc_topk_bboxes[g_id].unsqueeze(0),
                            dec_out_bboxes_gid
                        ])
                        out_logits_gid = paddle.concat([
                            enc_topk_logits[g_id].unsqueeze(0),
                            dec_out_logits_gid
                        ])
                        loss_gid = self.loss(
                            out_bboxes_gid,
                            out_logits_gid,
                            inputs['gt_bbox'],
                            inputs['gt_class'],
                            dn_out_bboxes=dn_out_bboxes_gid,
                            dn_out_logits=dn_out_logits_gid,
                            dn_meta=dn_meta[g_id])
                        # sum loss
                        for key, value in loss_gid.items():
                            loss.update({
                                key: loss.get(key, paddle.zeros([1])) + value
                            })

                    # average across (dual_groups + 1)
                    for key, value in loss.items():
                        loss.update({key: value / (dual_groups + 1)})
                    return loss
                else:
                    if self.group:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0],dn_meta['dn_num_split'][1] * 3]
                    elif self.groupx3_v3:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 3]
                    elif self.groupx3_v4:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 3]
                    elif self.groupx3_v5:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 3]
                    elif self.groupx3_v6:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 3]
                    elif self.groupx4:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 4]
                    elif self.groupx5:
                        dn_meta['dn_num_split'] = [dn_meta['dn_num_split'][0], dn_meta['dn_num_split'][1] * 5]
                    dn_out_bboxes, dec_out_bboxes = paddle.split(
                        dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                    dn_out_logits, dec_out_logits = paddle.split(
                        dec_out_logits, dn_meta['dn_num_split'], axis=2)


            else:
                dn_out_bboxes, dn_out_logits = None, None

            out_bboxes = paddle.concat(
                [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
            out_logits = paddle.concat(
                [enc_topk_logits.unsqueeze(0), dec_out_logits])

            if self.groupx5:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    dec_out=dec_out)
            elif self.groupx3_v3:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    all_feats=all_feats)
            elif self.groupx3_v4:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    all_feats=all_feats,
                    inputs=inputs,
                    generate_vis_pic=generate_vis_pic,
                    generate_ir_pic=generate_ir_pic)
            elif self.groupx3_v5:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    all_feats=all_feats,
                    inputs=inputs,
                    generate_vis_pic=generate_vis_pic,
                    generate_ir_pic=generate_ir_pic)
            elif self.groupx3_v6:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta,
                    all_feats=all_feats,
                    inputs=inputs,
                    generate_vis_pic=generate_vis_pic,
                    generate_ir_pic=generate_ir_pic)
            else:
                return self.loss(
                    out_bboxes,
                    out_logits,
                    inputs['gt_bbox'],
                    inputs['gt_class'],
                    dn_out_bboxes=dn_out_bboxes,
                    dn_out_logits=dn_out_logits,
                    dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], None)


@register
class MaskDINOHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self, loss='DINOLoss'):
        super(MaskDINOHead, self).__init__()
        self.loss = loss

    def forward(self, out_transformer, body_feats, inputs=None):
        (dec_out_logits, dec_out_bboxes, dec_out_masks, enc_out, init_out,
         dn_meta) = out_transformer
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            assert 'gt_segm' in inputs

            if dn_meta is not None:
                dn_out_logits, dec_out_logits = paddle.split(
                    dec_out_logits, dn_meta['dn_num_split'], axis=2)
                dn_out_bboxes, dec_out_bboxes = paddle.split(
                    dec_out_bboxes, dn_meta['dn_num_split'], axis=2)
                dn_out_masks, dec_out_masks = paddle.split(
                    dec_out_masks, dn_meta['dn_num_split'], axis=2)
                if init_out is not None:
                    init_out_logits, init_out_bboxes, init_out_masks = init_out
                    init_out_logits_dn, init_out_logits = paddle.split(
                        init_out_logits, dn_meta['dn_num_split'], axis=1)
                    init_out_bboxes_dn, init_out_bboxes = paddle.split(
                        init_out_bboxes, dn_meta['dn_num_split'], axis=1)
                    init_out_masks_dn, init_out_masks = paddle.split(
                        init_out_masks, dn_meta['dn_num_split'], axis=1)

                    dec_out_logits = paddle.concat(
                        [init_out_logits.unsqueeze(0), dec_out_logits])
                    dec_out_bboxes = paddle.concat(
                        [init_out_bboxes.unsqueeze(0), dec_out_bboxes])
                    dec_out_masks = paddle.concat(
                        [init_out_masks.unsqueeze(0), dec_out_masks])

                    dn_out_logits = paddle.concat(
                        [init_out_logits_dn.unsqueeze(0), dn_out_logits])
                    dn_out_bboxes = paddle.concat(
                        [init_out_bboxes_dn.unsqueeze(0), dn_out_bboxes])
                    dn_out_masks = paddle.concat(
                        [init_out_masks_dn.unsqueeze(0), dn_out_masks])
            else:
                dn_out_bboxes, dn_out_logits = None, None
                dn_out_masks = None

            enc_out_logits, enc_out_bboxes, enc_out_masks = enc_out
            out_logits = paddle.concat(
                [enc_out_logits.unsqueeze(0), dec_out_logits])
            out_bboxes = paddle.concat(
                [enc_out_bboxes.unsqueeze(0), dec_out_bboxes])
            out_masks = paddle.concat(
                [enc_out_masks.unsqueeze(0), dec_out_masks])

            return self.loss(
                out_bboxes,
                out_logits,
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=out_masks,
                gt_mask=inputs['gt_segm'],
                dn_out_logits=dn_out_logits,
                dn_out_bboxes=dn_out_bboxes,
                dn_out_masks=dn_out_masks,
                dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], dec_out_masks[-1])
