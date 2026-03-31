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
from ppdet.core.workspace import register
from .iou_loss import GIoULoss
from ..transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss, varifocal_loss_with_logits
from ..bbox_utils import bbox_iou
from ext_op import matched_rbox_iou
from ..losses.probiou_loss import ProbIoULoss
__all__ = [
    'DETRLoss', 'DINOLoss', 'DINOLoss_Rotate', 'DINOLoss_Rotate_RouteROI',
    'DETRLoss_Rotate'
]


@register
class DETRLoss_Rotate(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher_Rotate',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 group=False,
                 groupx4=False,
                 groupx5=False,
                 groupx3_v3=False,
                 groupx3_v4=False,
                 groupx3_v5=False,
                 groupx3_v6=False,
                 uni_match_ind=0,
                 angle_ignore_short_side=2.0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss_Rotate, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.group = group
        self.groupx4 = groupx4
        self.groupx5 = groupx5
        self.groupx3_v3 = groupx3_v3
        self.groupx3_v4 = groupx3_v4
        self.groupx3_v5 = groupx3_v5
        self.groupx3_v6 = groupx3_v6
        self.angle_ignore_short_side = float(angle_ignore_short_side)
        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.angle_max = 90
        self.half_pi_bin = self.half_pi / self.angle_max

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        #self.giou_loss = GIoULoss()
        self.piou_loss = ProbIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = paddle.zeros([bs, num_query_objects])
                if num_gt > 0:
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, iou_score)
                target_score = target_score.reshape(
                    [bs, num_query_objects, 1]) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                    logits, target_score, target_label,
                    num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, angle_cls, gt_bbox, match_indices, num_gts, im_shape,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_piou" + postfix
        name_angle = "loss_angle" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            loss[name_angle] = paddle.to_tensor([0.])
            return loss

        src_bbox, src_angle_cls, target_bbox = self._get_src_target_assign_r(boxes, angle_cls, gt_bbox,
                                                            match_indices)

        target_bbox_normalize = target_bbox[:, :4] / im_shape
        src_bbox_real = src_bbox[:, :4] * im_shape
        src_bbox_real = paddle.concat([src_bbox_real, src_bbox[:, -1].unsqueeze(axis=1)], axis=-1)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox[:, :4], target_bbox_normalize, reduction='sum') / num_gts
        loss[name_giou] = self.piou_loss(
            src_bbox_real, target_bbox)
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]

        tgt_angle_pos = (
                target_bbox[:, 4] /
                self.half_pi_bin).clip(0, self.angle_max - 0.01)
        angle_loss = self._df_loss(src_angle_cls, tgt_angle_pos)
        short_side = paddle.minimum(target_bbox[:, 2], target_bbox[:, 3]).reshape([-1, 1])
        valid_mask = (short_side >= self.angle_ignore_short_side).astype(angle_loss.dtype)
        valid_count = paddle.sum(valid_mask)
        loss[name_angle] = paddle.sum(angle_loss * valid_mask) / paddle.maximum(
            valid_count, paddle.ones_like(valid_count))
        loss[name_angle] = loss[name_angle] * 0.05

        return loss


    @staticmethod
    def _df_loss(pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none')
        loss_left = loss_left * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      angle_cls,
                      gt_bbox,
                      gt_class,
                      im_shape,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        loss_angle = []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_angle_cls, aux_logits) in enumerate(zip(boxes, angle_cls, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    aux_angle_cls,
                    gt_bbox,
                    gt_class,
                    im_shape,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            #im_shape_ = np.array(im_shape)
            im_shape_ = paddle.tile(im_shape, [2])
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    src_bbox_real = src_bbox[:, :4] * im_shape_
                    src_bbox_real = paddle.concat([src_bbox_real, src_bbox[:, -1].unsqueeze(axis=1)], axis=-1)
                    iou_score = matched_rbox_iou(
                        src_bbox_real,
                        target_bbox).unsqueeze(1)

                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes,aux_angle_cls, gt_bbox, match_indices,
                                        num_gts,im_shape_, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_piou' + postfix])
            loss_angle.append(loss_['loss_angle' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_piou_aux" + postfix: paddle.add_n(loss_giou),
            "loss_angle_aux" + postfix: paddle.add_n(loss_angle)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign_r(self, src, src_angle, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        src_assign_angle = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src_angle, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, src_assign_angle, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             angles_cls,
                             gt_bbox,
                             gt_class,
                             im_shape,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, angles_cls, gt_bbox, gt_class,im_shape, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        #im_shape = np.array(im_shape)
        im_shape = paddle.tile(im_shape, [2])



        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)

                src_bbox_real = src_bbox[:, :4] * im_shape
                src_bbox_real = paddle.concat([src_bbox_real, src_bbox[:, -1].unsqueeze(axis=1)], axis=-1)

                iou_score = matched_rbox_iou(
                    src_bbox_real,
                    target_bbox).unsqueeze(1)
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, angles_cls, gt_bbox, match_indices, num_gts,im_shape,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                angles_cls,
                gt_bbox,
                gt_class,
                im_shape,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            angles_cls[-1],
            gt_bbox,
            gt_class,
            im_shape,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    angles_cls[:-1],
                    gt_bbox,
                    gt_class,
                    im_shape,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss
@register
class DINOLoss_Rotate(DETRLoss_Rotate):
    def forward(self,
                boxes,
                logits,
                angles_cls,
                gt_bbox,
                gt_class,
                im_shape,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_angle_cls=None,
                dn_meta=None,
                dec_out=None,
                all_feats=None,
                inputs=None,
                generate_vis_pic=None,
                generate_ir_pic=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        num_quires = 300
        if self.group:
            total_loss = {}
            boxes = paddle.split(boxes,3,axis=2)
            logits = paddle.split(logits,3,axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
        elif self.groupx5:
            total_loss = {}
            dec_out_loss_vis = 0
            dec_out_loss_ir = 0
            # dec_out loss
            for dec_out_l in dec_out:
                dec_out_l_f_vis = dec_out_l[:,-num_quires*4:-num_quires*3,:]
                dec_out_l_f_ir = dec_out_l[:,-num_quires*3:-num_quires*2,:]
                dec_out_l_vis = dec_out_l[:,-num_quires*2:-num_quires,:]
                dec_out_l_ir = dec_out_l[:,-num_quires:,:]
                dec_out_loss_vis = dec_out_loss_vis + paddle.nn.functional.mse_loss(dec_out_l_f_vis,dec_out_l_vis,reduction='mean')
                dec_out_loss_ir = dec_out_loss_ir + paddle.nn.functional.mse_loss(dec_out_l_f_ir,dec_out_l_ir,reduction='mean')


            boxes = paddle.split(boxes, 5, axis=2)
            logits = paddle.split(logits, 5, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            total_loss4 = super(DINOLoss, self).forward(
                boxes[3], logits[3], gt_bbox, gt_class, num_gts=num_gts)
            total_loss5 = super(DINOLoss, self).forward(
                boxes[4], logits[4], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]+ total_loss4[key] + total_loss5[key]) / 5
            total_loss['dec_out_loss_vis'] = dec_out_loss_vis
            total_loss['dec_out_loss_ir'] = dec_out_loss_ir
        elif self.groupx3_v3:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
            vis_feats = all_feats[:3]
            ir_feats = all_feats[3:6]
            vis_feats_g = all_feats[6:9]
            ir_feats_g = all_feats[9:12]
            for mm in range(3):
                vis_feats_d = vis_feats[mm].detach()
                ir_feats_d = ir_feats[mm].detach()
                # vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_d, vis_feats_g[mm],
                #                                                         reduction='mean')
                # ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_d, ir_feats_g[mm],
                #                                                       reduction='mean')
                vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_g[mm],vis_feats_d,
                                                                        reduction='mean')
                ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss( ir_feats_g[mm],ir_feats_d,
                                                                      reduction='mean')


            total_loss['vis_g_loss'] = vis_g_loss
            total_loss['ir_g_loss'] = ir_g_loss

        elif self.groupx3_v4:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
            # vis_feats = all_feats[:3]
            # ir_feats = all_feats[3:6]
            # vis_feats_g = all_feats[6:9]
            # ir_feats_g = all_feats[9:12]
            # for mm in range(3):
            #     vis_feats_d = vis_feats[mm].detach()
            #     ir_feats_d = ir_feats[mm].detach()
            #     # vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_d, vis_feats_g[mm],
            #     #                                                         reduction='mean')
            #     # ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_d, ir_feats_g[mm],
            #     #                                                       reduction='mean')
            #     vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_g[mm],vis_feats_d,
            #                                                             reduction='mean')
            #     ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_g[mm],ir_feats_d,
            #                                                           reduction='mean')
            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.mse_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.mse_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_loss'] = ir_g_pic_loss * 10

        elif self.groupx3_v5:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3

            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.l1_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.l1_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_l1loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_l1loss'] = ir_g_pic_loss * 10

        elif self.groupx3_v6:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3

            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.l1_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.l1_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_l1loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_l1loss'] = ir_g_pic_loss * 10

        elif self.groupx4:
            total_loss = {}

            boxes = paddle.split(boxes, 4, axis=2)
            logits = paddle.split(logits, 4, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            total_loss4 = super(DINOLoss, self).forward(
                boxes[3], logits[3], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (
                            total_loss1[key] * 0.4 + total_loss2[key] * 0.2 + total_loss3[key] * 0.2 + total_loss4[
                        key] * 0.2)

        else:
            total_loss = super(DINOLoss_Rotate, self).forward(
                boxes, logits,angles_cls, gt_bbox, gt_class,im_shape, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss_Rotate, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                dn_out_angle_cls,
                gt_bbox,
                gt_class,
                im_shape,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices


@register
class DINOLoss_Rotate_RouteROI(DINOLoss_Rotate):
    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher_Rotate',
                 loss_coeff=None,
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 group=False,
                 groupx4=False,
                 groupx5=False,
                 groupx3_v3=False,
                 groupx3_v4=False,
                 groupx3_v5=False,
                 groupx3_v6=False,
                 uni_match_ind=0,
                 angle_ignore_short_side=2.0):
        loss_coeff = loss_coeff or {
            'class': 1,
            'bbox': 5,
            'giou': 2,
            'no_object': 0.1,
            'route_primary': 1.0,
            'visible': 1.0,
            'cam_class': 1.0,
            'proj_l1': 2.0,
            'proj_iou': 2.0,
        }
        super(DINOLoss_Rotate_RouteROI, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            loss_coeff=loss_coeff,
            aux_loss=aux_loss,
            use_focal_loss=use_focal_loss,
            use_vfl=use_vfl,
            use_uni_match=use_uni_match,
            group=group,
            groupx4=groupx4,
            groupx5=groupx5,
            groupx3_v3=groupx3_v3,
            groupx3_v4=groupx3_v4,
            groupx3_v5=groupx3_v5,
            groupx3_v6=groupx3_v6,
            uni_match_ind=uni_match_ind,
            angle_ignore_short_side=angle_ignore_short_side)

    @staticmethod
    def _meta_to_list(value):
        if isinstance(value, (list, tuple)):
            return [
                item if isinstance(item, paddle.Tensor) else paddle.to_tensor(item)
                for item in value
            ]
        if isinstance(value, paddle.Tensor):
            return [value[i] for i in range(value.shape[0])]
        return [paddle.to_tensor(value)]

    @staticmethod
    def _box_iou_xyxy(box1, box2):
        lt = paddle.maximum(box1[:, :2], box2[:, :2])
        rb = paddle.minimum(box1[:, 2:], box2[:, 2:])
        wh = paddle.clip(rb - lt, min=0.0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = paddle.clip(box1[:, 2] - box1[:, 0], min=0.0) * paddle.clip(
            box1[:, 3] - box1[:, 1], min=0.0)
        area2 = paddle.clip(box2[:, 2] - box2[:, 0], min=0.0) * paddle.clip(
            box2[:, 3] - box2[:, 1], min=0.0)
        union = paddle.clip(area1 + area2 - inter, min=1e-6)
        return inter / union

    def _get_route_roi_losses(self,
                              boxes,
                              logits,
                              angles_cls,
                              gt_bbox,
                              gt_class,
                              im_shape,
                              route_primary_logits,
                              visible_logits,
                              camera_cls_logits,
                              camera_refined_boxes,
                              inputs):
        zeros = paddle.sum(boxes) * 0.0
        match_indices = self.matcher(boxes, logits, angles_cls, gt_bbox,
                                     gt_class, im_shape)

        gt_primary_camera = self._meta_to_list(inputs['gt_primary_camera'])
        gt_visible_cameras = self._meta_to_list(inputs['gt_visible_cameras'])
        gt_camera_box_2d = self._meta_to_list(inputs['gt_camera_box_2d'])
        gt_has_camera_box = self._meta_to_list(inputs['gt_has_camera_box'])
        camera_img_size = paddle.to_tensor(inputs['camera_img_size'], dtype='float32')
        if len(camera_img_size.shape) == 2:
            camera_img_size = camera_img_size.unsqueeze(0)

        route_logits_all = []
        route_targets_all = []
        visible_logits_all = []
        visible_targets_all = []
        camera_logits_all = []
        camera_targets_all = []
        proj_box_pred_all = []
        proj_box_tgt_all = []
        proj_box_wh_all = []

        for batch_id, (src_idx, dst_idx) in enumerate(match_indices):
            if src_idx.shape[0] == 0:
                continue

            primary = paddle.gather(
                gt_primary_camera[batch_id].reshape([-1]).astype('int64'),
                dst_idx,
                axis=0)
            visible = paddle.gather(
                gt_visible_cameras[batch_id].astype('float32'),
                dst_idx,
                axis=0)
            route_logits_all.append(route_primary_logits[batch_id, src_idx])
            route_targets_all.append(primary)
            visible_logits_all.append(visible_logits[batch_id, src_idx])
            visible_targets_all.append(visible)

            valid_primary = primary < 4
            if paddle.any(valid_primary):
                valid_idx = paddle.nonzero(valid_primary).flatten()
                src_valid = paddle.gather(src_idx, valid_idx, axis=0)
                dst_valid = paddle.gather(dst_idx, valid_idx, axis=0)
                primary_valid = paddle.gather(primary, valid_idx, axis=0)

                target_boxes_by_cam = paddle.gather(
                    gt_camera_box_2d[batch_id].astype('float32'),
                    dst_valid,
                    axis=0)
                target_has_box_by_cam = paddle.gather(
                    gt_has_camera_box[batch_id].astype('float32'),
                    dst_valid,
                    axis=0)
                gather_cam_idx = paddle.stack(
                    [paddle.arange(primary_valid.shape[0], dtype='int64'),
                     primary_valid.astype('int64')],
                    axis=-1)
                target_box = paddle.gather_nd(target_boxes_by_cam, gather_cam_idx)
                target_has_box = paddle.gather_nd(target_has_box_by_cam,
                                                  gather_cam_idx) > 0.5

                if paddle.any(target_has_box):
                    camera_valid_idx = paddle.nonzero(target_has_box).flatten()
                    src_cam = paddle.gather(src_valid, camera_valid_idx, axis=0)
                    dst_cam = paddle.gather(dst_valid, camera_valid_idx, axis=0)
                    cam_id = paddle.gather(primary_valid, camera_valid_idx, axis=0)
                    gather_cam_idx = paddle.stack(
                        [paddle.arange(cam_id.shape[0], dtype='int64'),
                         cam_id.astype('int64')],
                        axis=-1)

                    camera_logits = paddle.gather_nd(
                        camera_cls_logits[batch_id, src_cam], gather_cam_idx)
                    class_targets = paddle.gather(
                        gt_class[batch_id].reshape([-1]).astype('int64'),
                        dst_cam,
                        axis=0)

                    pred_box = paddle.gather_nd(
                        camera_refined_boxes[batch_id, src_cam], gather_cam_idx)
                    tgt_box = paddle.gather_nd(
                        paddle.gather(target_boxes_by_cam, camera_valid_idx, axis=0),
                        gather_cam_idx)
                    wh = paddle.gather(
                        camera_img_size[batch_id],
                        cam_id.astype('int64'),
                        axis=0)
                    whxy = paddle.concat(
                        [wh[:, 0:1], wh[:, 1:2], wh[:, 0:1], wh[:, 1:2]], axis=-1)

                    camera_logits_all.append(camera_logits)
                    camera_targets_all.append(class_targets)
                    proj_box_pred_all.append(pred_box)
                    proj_box_tgt_all.append(tgt_box)
                    proj_box_wh_all.append(whxy)

        loss = {
            'loss_route_primary': zeros,
            'loss_visible': zeros,
            'loss_cam_class': zeros,
            'loss_proj_l1': zeros,
            'loss_proj_iou': zeros,
        }

        if route_logits_all:
            route_logits_all = paddle.concat(route_logits_all, axis=0)
            route_targets_all = paddle.concat(route_targets_all, axis=0)
            loss['loss_route_primary'] = F.cross_entropy(
                route_logits_all, route_targets_all, reduction='mean')
            loss['loss_route_primary'] *= self.loss_coeff.get(
                'route_primary', 1.0)

        if visible_logits_all:
            visible_logits_all = paddle.concat(visible_logits_all, axis=0)
            visible_targets_all = paddle.concat(visible_targets_all, axis=0)
            loss['loss_visible'] = F.binary_cross_entropy_with_logits(
                visible_logits_all, visible_targets_all, reduction='mean')
            loss['loss_visible'] *= self.loss_coeff.get('visible', 1.0)

        if camera_logits_all:
            camera_logits_all = paddle.concat(camera_logits_all, axis=0)
            camera_targets_all = paddle.concat(camera_targets_all, axis=0)
            proj_box_pred_all = paddle.concat(proj_box_pred_all, axis=0)
            proj_box_tgt_all = paddle.concat(proj_box_tgt_all, axis=0)
            proj_box_wh_all = paddle.concat(proj_box_wh_all, axis=0)
            loss['loss_cam_class'] = F.cross_entropy(
                camera_logits_all, camera_targets_all, reduction='mean')
            loss['loss_cam_class'] *= self.loss_coeff.get('cam_class', 1.0)
            loss['loss_proj_l1'] = F.smooth_l1_loss(
                proj_box_pred_all / paddle.clip(proj_box_wh_all, min=1e-6),
                proj_box_tgt_all / paddle.clip(proj_box_wh_all, min=1e-6),
                reduction='mean')
            loss['loss_proj_l1'] *= self.loss_coeff.get('proj_l1', 2.0)
            iou = self._box_iou_xyxy(proj_box_pred_all, proj_box_tgt_all)
            loss['loss_proj_iou'] = (1.0 - iou).mean()
            loss['loss_proj_iou'] *= self.loss_coeff.get('proj_iou', 2.0)

        return loss

    def forward(self,
                boxes,
                logits,
                angles_cls,
                gt_bbox,
                gt_class,
                im_shape,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_angle_cls=None,
                dn_meta=None,
                **kwargs):
        total_loss = super(DINOLoss_Rotate_RouteROI, self).forward(
            boxes,
            logits,
            angles_cls,
            gt_bbox,
            gt_class,
            im_shape,
            masks=masks,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_out_bboxes=dn_out_bboxes,
            dn_out_logits=dn_out_logits,
            dn_out_angle_cls=dn_out_angle_cls,
            dn_meta=dn_meta,
            **kwargs)

        route_primary_logits = kwargs.get('route_primary_logits', None)
        visible_logits = kwargs.get('visible_logits', None)
        camera_cls_logits = kwargs.get('camera_cls_logits', None)
        camera_refined_boxes = kwargs.get('camera_refined_boxes', None)
        inputs = kwargs.get('inputs', None)
        if route_primary_logits is None or visible_logits is None or \
                camera_cls_logits is None or camera_refined_boxes is None or inputs is None:
            return total_loss

        extra_loss = self._get_route_roi_losses(
            boxes[-1], logits[-1], angles_cls[-1], gt_bbox, gt_class, im_shape,
            route_primary_logits, visible_logits, camera_cls_logits,
            camera_refined_boxes, inputs)
        total_loss.update(extra_loss)
        return total_loss
@register
class DETRLoss(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 group=False,
                 groupx4=False,
                 groupx5=False,
                 groupx3_v3=False,
                 groupx3_v4=False,
                 groupx3_v5=False,
                 groupx3_v6=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.group = group
        self.groupx4 = groupx4
        self.groupx5 = groupx5
        self.groupx3_v3 = groupx3_v3
        self.groupx3_v4 = groupx3_v4
        self.groupx3_v5 = groupx3_v5
        self.groupx3_v6 = groupx3_v6

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = paddle.zeros([bs, num_query_objects])
                if num_gt > 0:
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, iou_score)
                target_score = target_score.reshape(
                    [bs, num_query_objects, 1]) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                    logits, target_score, target_label,
                    num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                    bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss


@register
class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                dec_out=None,
                all_feats=None,
                inputs=None,
                generate_vis_pic=None,
                generate_ir_pic=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        num_quires = 300
        if self.group:
            total_loss = {}
            boxes = paddle.split(boxes,3,axis=2)
            logits = paddle.split(logits,3,axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
        elif self.groupx5:
            total_loss = {}
            dec_out_loss_vis = 0
            dec_out_loss_ir = 0
            # dec_out loss
            for dec_out_l in dec_out:
                dec_out_l_f_vis = dec_out_l[:,-num_quires*4:-num_quires*3,:]
                dec_out_l_f_ir = dec_out_l[:,-num_quires*3:-num_quires*2,:]
                dec_out_l_vis = dec_out_l[:,-num_quires*2:-num_quires,:]
                dec_out_l_ir = dec_out_l[:,-num_quires:,:]
                dec_out_loss_vis = dec_out_loss_vis + paddle.nn.functional.mse_loss(dec_out_l_f_vis,dec_out_l_vis,reduction='mean')
                dec_out_loss_ir = dec_out_loss_ir + paddle.nn.functional.mse_loss(dec_out_l_f_ir,dec_out_l_ir,reduction='mean')


            boxes = paddle.split(boxes, 5, axis=2)
            logits = paddle.split(logits, 5, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            total_loss4 = super(DINOLoss, self).forward(
                boxes[3], logits[3], gt_bbox, gt_class, num_gts=num_gts)
            total_loss5 = super(DINOLoss, self).forward(
                boxes[4], logits[4], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]+ total_loss4[key] + total_loss5[key]) / 5
            total_loss['dec_out_loss_vis'] = dec_out_loss_vis
            total_loss['dec_out_loss_ir'] = dec_out_loss_ir
        elif self.groupx3_v3:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
            vis_feats = all_feats[:3]
            ir_feats = all_feats[3:6]
            vis_feats_g = all_feats[6:9]
            ir_feats_g = all_feats[9:12]
            for mm in range(3):
                vis_feats_d = vis_feats[mm].detach()
                ir_feats_d = ir_feats[mm].detach()
                # vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_d, vis_feats_g[mm],
                #                                                         reduction='mean')
                # ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_d, ir_feats_g[mm],
                #                                                       reduction='mean')
                vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_g[mm],vis_feats_d,
                                                                        reduction='mean')
                ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss( ir_feats_g[mm],ir_feats_d,
                                                                      reduction='mean')


            total_loss['vis_g_loss'] = vis_g_loss
            total_loss['ir_g_loss'] = ir_g_loss

        elif self.groupx3_v4:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3
            # vis_feats = all_feats[:3]
            # ir_feats = all_feats[3:6]
            # vis_feats_g = all_feats[6:9]
            # ir_feats_g = all_feats[9:12]
            # for mm in range(3):
            #     vis_feats_d = vis_feats[mm].detach()
            #     ir_feats_d = ir_feats[mm].detach()
            #     # vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_d, vis_feats_g[mm],
            #     #                                                         reduction='mean')
            #     # ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_d, ir_feats_g[mm],
            #     #                                                       reduction='mean')
            #     vis_g_loss = vis_g_loss + paddle.nn.functional.mse_loss(vis_feats_g[mm],vis_feats_d,
            #                                                             reduction='mean')
            #     ir_g_loss = ir_g_loss + paddle.nn.functional.mse_loss(ir_feats_g[mm],ir_feats_d,
            #                                                           reduction='mean')
            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.mse_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.mse_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_loss'] = ir_g_pic_loss * 10

        elif self.groupx3_v5:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3

            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.l1_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.l1_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_l1loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_l1loss'] = ir_g_pic_loss * 10

        elif self.groupx3_v6:
            total_loss = {}
            vis_g_loss = 0
            ir_g_loss = 0
            boxes = paddle.split(boxes, 3, axis=2)
            logits = paddle.split(logits, 3, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (total_loss1[key] + total_loss2[key] + total_loss3[key]) / 3

            gt_vis_pic = inputs['vis_image']
            gt_ir_pic = inputs['ir_image']
            vis_g_pic_loss = paddle.nn.functional.l1_loss(generate_vis_pic,gt_vis_pic ,
                                                                      reduction='mean')
            ir_g_pic_loss = paddle.nn.functional.l1_loss(generate_ir_pic,gt_ir_pic ,
                                                                      reduction='mean')

            total_loss['vis_g_pic_l1loss'] = vis_g_pic_loss * 10
            total_loss['ir_g_pic_l1loss'] = ir_g_pic_loss * 10

        elif self.groupx4:
            total_loss = {}

            boxes = paddle.split(boxes, 4, axis=2)
            logits = paddle.split(logits, 4, axis=2)
            total_loss1 = super(DINOLoss, self).forward(
                boxes[0], logits[0], gt_bbox, gt_class, num_gts=num_gts)
            total_loss2 = super(DINOLoss, self).forward(
                boxes[1], logits[1], gt_bbox, gt_class, num_gts=num_gts)
            total_loss3 = super(DINOLoss, self).forward(
                boxes[2], logits[2], gt_bbox, gt_class, num_gts=num_gts)
            total_loss4 = super(DINOLoss, self).forward(
                boxes[3], logits[3], gt_bbox, gt_class, num_gts=num_gts)
            for key in total_loss1:
                total_loss[key] = (
                            total_loss1[key] * 0.4 + total_loss2[key] * 0.2 + total_loss3[key] * 0.2 + total_loss4[
                        key] * 0.2)

        else:
            total_loss = super(DINOLoss, self).forward(
                boxes, logits, gt_bbox, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices


@register
class MaskDINOLoss(DETRLoss):
    __shared__ = ['num_classes', 'use_focal_loss', 'num_sample_points']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 4,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 5,
                     'dice': 5
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75):
        super(MaskDINOLoss, self).__init__(num_classes, matcher, loss_coeff,
                                           aux_loss, use_focal_loss)
        assert oversample_ratio >= 1
        assert important_sample_ratio <= 1 and important_sample_ratio >= 0

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points *
                                        important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(MaskDINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            masks=masks,
            gt_mask=gt_mask,
            num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = DINOLoss.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(MaskDINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                masks=dn_out_masks,
                gt_mask=gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        # sample points
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

        src_masks = F.grid_sample(
            src_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2])

        target_masks = F.grid_sample(
            target_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2]).detach()

        loss[name_mask] = self.loss_coeff[
            'mask'] * F.binary_cross_entropy_with_logits(
                src_masks, target_masks,
                reduction='none').mean(1).sum() / num_gts
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _get_point_coords_by_uncertainty(self, masks):
        # Sample points based on their uncertainty.
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = paddle.rand(
            [num_masks, 1, self.num_oversample_points, 2])

        out_mask = F.grid_sample(
            masks.unsqueeze(1), 2.0 * sample_points - 1.0,
            align_corners=False).squeeze([1, 2])
        out_mask = -paddle.abs(out_mask)

        _, topk_ind = paddle.topk(out_mask, self.num_important_points, axis=1)
        batch_ind = paddle.arange(end=num_masks, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_important_points])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        sample_points = paddle.gather_nd(sample_points.squeeze(1), topk_ind)
        if self.num_random_points > 0:
            sample_points = paddle.concat(
                [
                    sample_points,
                    paddle.rand([num_masks, self.num_random_points, 2])
                ],
                axis=1)
        return sample_points
