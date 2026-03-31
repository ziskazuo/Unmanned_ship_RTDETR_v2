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
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['RadarCamera_DETR', 'RadarCamera_RouteROI_DETR']
# My multispectral DETR


@register
class RadarCamera_DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_radar,
                 backbone_camera,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_radar=None,
                 neck_camera=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(RadarCamera_DETR, self).__init__()
        self.backbone_radar = backbone_radar
        self.backbone_camera = backbone_camera
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_radar = neck_radar
        self.neck_camera = neck_camera
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_radar = create(cfg['backbone_radar'])
        # backbone_ir
        backbone_camera = create(cfg['backbone_camera'])
        # neck
        kwargs = {'input_shape': backbone_radar.out_shape}
        neck_radar = create(cfg['neck_radar'], **kwargs) if cfg['neck_radar'] else None
        neck_camera = create(cfg['neck_camera'], **kwargs) if cfg['neck_camera'] else None

        # transformer
        if neck_radar is not None:
            kwargs = {'input_shape': neck_radar.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_radar.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_radar': backbone_radar,
            'backbone_camera': backbone_camera,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_radar": neck_radar,
            "neck_camera": neck_camera
        }

    def _forward(self):
        if self.training == False:
            self.inputs['camera_image'] = paddle.stack(self.inputs['camera_image'], axis=1)

        # Backbone
        radar_body_feats = self.backbone_radar(self.inputs,'radar')        
        camera_body_feats = self.backbone_camera(self.inputs,'camera')
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])

        # Neck
        if self.neck_radar is not None:
            #body_feats = self.neck(body_feats)
            radar_body_feats = self.neck_radar(radar_body_feats)
            camera_body_feats = self.neck_camera(camera_body_feats)

        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        out_transformer = self.transformer(None,radar_body_feats, camera_body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, None)
            #preds = (preds[0][:, :900,:],preds[1][:,600:900,:],preds[2])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['radar_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


@register
class RadarCamera_RouteROI_DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_radar,
                 backbone_camera,
                 transformer='RTDETRTransformer_Rotate_RouteROI',
                 detr_head='DINOHead_Rotate_RouteROI',
                 neck_radar=None,
                 neck_camera=None,
                 post_process='DETRPostProcess_Rotate',
                 with_mask=False,
                 exclude_post_process=False):
        super(RadarCamera_RouteROI_DETR, self).__init__()
        self.backbone_radar = backbone_radar
        self.backbone_camera = backbone_camera
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_radar = neck_radar
        self.neck_camera = neck_camera
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone_radar = create(cfg['backbone_radar'])
        backbone_camera = create(cfg['backbone_camera'])

        radar_kwargs = {'input_shape': backbone_radar.out_shape}
        camera_kwargs = {'input_shape': backbone_camera.out_shape}
        neck_radar = create(
            cfg['neck_radar'], **radar_kwargs) if cfg['neck_radar'] else None
        neck_camera = create(
            cfg['neck_camera'],
            **camera_kwargs) if cfg['neck_camera'] else None

        transformer_input_shape = neck_radar.out_shape if neck_radar is not None else backbone_radar.out_shape
        transformer = create(
            cfg['transformer'], input_shape=transformer_input_shape)

        head_input_shape = neck_camera.out_shape if neck_camera is not None else backbone_camera.out_shape
        detr_head = create(
            cfg['detr_head'],
            hidden_dim=transformer.hidden_dim,
            nhead=transformer.nhead,
            input_shape=head_input_shape)

        return {
            'backbone_radar': backbone_radar,
            'backbone_camera': backbone_camera,
            'transformer': transformer,
            'detr_head': detr_head,
            'neck_radar': neck_radar,
            'neck_camera': neck_camera,
        }

    def _forward(self):
        if self.training == False:
            self.inputs['camera_image'] = paddle.stack(
                self.inputs['camera_image'], axis=1)

        radar_body_feats = self.backbone_radar(self.inputs, 'radar')
        camera_body_feats = self.backbone_camera(self.inputs, 'camera')

        if self.neck_radar is not None:
            radar_body_feats = self.neck_radar(radar_body_feats)
        if self.neck_camera is not None:
            camera_body_feats = self.neck_camera(camera_body_feats)

        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(
            radar_body_feats, pad_mask, self.inputs)

        if self.training:
            detr_losses = self.detr_head(out_transformer, camera_body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses

        preds = self.detr_head(out_transformer, camera_body_feats, self.inputs)
        if self.exclude_post_process:
            bbox, bbox_num, mask = preds
        else:
            bbox, bbox_num, mask = self.post_process(
                preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                paddle.shape(self.inputs['radar_image'])[2:])

        output = {'bbox': bbox, 'bbox_num': bbox_num}
        if self.with_mask:
            output['mask'] = mask
        return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
