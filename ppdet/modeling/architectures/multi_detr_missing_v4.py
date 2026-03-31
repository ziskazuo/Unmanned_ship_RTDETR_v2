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

import numpy
import numpy as np
import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
import paddle.nn as nn
import paddle.nn.functional as F
import cv2

__all__ = ['Multi_DETR_MISSING_V4','Complementary_Decoder2','Generator']
# My multispectral DETR


class ConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 act = None,
                 norm = 'Instance'):
        super(ConvNormLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            bias_attr=False)
        #self.norm = nn.BatchNorm2D(ch_out)
        if norm == 'Instance':
            self.norm = nn.InstanceNorm2D(ch_out)
        elif norm == 'Batch':
            self.norm = nn.BatchNorm2D(ch_out)

    def forward(self,inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out

class DeConvNormLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 act = None,
                 norm = 'Instance'):
        super(DeConvNormLayer, self).__init__()
        self.act = act
        self.deconv = nn.Conv2DTranspose(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            bias_attr=False,
            dilation=1)
        #self.norm = nn.BatchNorm2D(ch_out)
        if norm == 'Instance':
            self.norm = nn.InstanceNorm2D(ch_out)
        elif norm == 'Batch':
            self.norm = nn.BatchNorm2D(ch_out)

    def forward(self,inputs):
        out = self.deconv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


@register
class Complementary_Decoder2(nn.Layer):
    def __init__(self,
                 ch_in1,
                 ch_in2,
                 ch_in3):
        super(Complementary_Decoder2, self).__init__()
        # self.res1 = ResBlock(ch_in3,ch_in3)
        # self.res2 = ResBlock(ch_in3,ch_in3)
        self.upsample = nn.UpsamplingBilinear2D(scale_factor=2)
        self.cov0_1 = ConvNormLayer(ch_in3,ch_in3,3,1,act='silu')
        self.cov0_2 = ConvNormLayer(ch_in3,ch_in3,3,1,act='silu')
        self.cov1_1 = ConvNormLayer(ch_in3+ch_in2,ch_in2,3,1,act='silu')
        self.cov1_2 = ConvNormLayer(ch_in2,ch_in2,3,1,act='silu')
        self.cov2_1 = ConvNormLayer(ch_in2+ch_in1,ch_in1,3,1,act='silu')
        self.cov2_2 = ConvNormLayer(ch_in1, ch_in1, 3, 1, act='silu')

    def forward(self,inputs):
        x0 = inputs[0]
        x1 = inputs[1]
        x2 = inputs[2]
        out0 = self.cov0_1(x2)
        out0 = self.cov0_2(out0)
        out1 = self.upsample(out0)
        out1 = paddle.concat([out1,x1],axis=1)
        out1 = self.cov1_1(out1)
        out1 = self.cov1_2(out1)
        out2 = self.upsample(out1)
        out2 = paddle.concat([out2,x0],axis=1)
        out2 = self.cov2_1(out2)
        out2 = self.cov2_2(out2)

        out = [out2,out1,out0]
        return out

@register
class Generator(nn.Layer):
    def __init__(self,
                 ch_in1,
                 ):
        super(Generator, self).__init__()
        # self.res1 = ResBlock(ch_in3,ch_in3)
        # self.res2 = ResBlock(ch_in3,ch_in3)
        self.upsample = nn.UpsamplingBilinear2D(scale_factor=2)
        self.cov1 = DeConvNormLayer(ch_in1,ch_in1//2,4,2,act='relu',norm='Batch')
        self.cov2 = DeConvNormLayer(ch_in1//2,ch_in1//4,4,2,act='relu',norm='Batch')
        self.cov3 = DeConvNormLayer(ch_in1//4,3,4,2,act='sigmoid',norm='Batch')

    def forward(self,inputs):
        x0 = inputs[0]
        x1 = inputs[1]
        x2 = inputs[2]
        x2 = self.upsample(x2)
        x1 = x1+x2
        x1 = self.upsample(x1)
        x = x0 + x1
        out = self.cov1(x)
        out = self.cov2(out)
        out = self.cov3(out)
        return out

@register
class Multi_DETR_MISSING_V4(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 complementary_g_forvis,
                 complementary_g_forir,
                 generator,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_vis=None,
                 neck_ir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(Multi_DETR_MISSING_V4, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.complementary_g_forvis = complementary_g_forvis
        self.complementary_g_forir = complementary_g_forir
        self.generator = generator
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_vis = neck_vis
        self.neck_ir = neck_ir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # complementary generator for vis
        complementary_g_forvis = create(cfg['complementary_g_forvis'])

        # complementary generator for ir
        complementary_g_forir = create(cfg['complementary_g_forir'])

        # generator
        generator = create(cfg['generator'])

        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_vis = create(cfg['neck_vis'], **kwargs) if cfg['neck_vis'] else None
        neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None

        # transformer
        if neck_vis is not None:
            kwargs = {'input_shape': neck_vis.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'complementary_g_forvis': complementary_g_forvis,
            'complementary_g_forir': complementary_g_forir,
            'generator': generator,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_vis": neck_vis,
            "neck_ir": neck_ir
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Neck
        if self.neck_vis is not None:
            #body_feats = self.neck(body_feats)
            vis_body_feats_temp = []
            ir_body_feats_temp = []
            vis_body_feats = self.neck_vis(vis_body_feats)
            ir_body_feats = self.neck_ir(ir_body_feats)
            for x in range(3):
                vis_body_feats_temp.append(vis_body_feats[x].detach())
                ir_body_feats_temp.append(ir_body_feats[x].detach())
            ir_body_feats_g = self.complementary_g_forvis(vis_body_feats_temp)
            vis_body_feats_g = self.complementary_g_forir(ir_body_feats_temp)

        #generate vis pic
        generate_vis_pic = self.generator(vis_body_feats_g)
        #generate ir pic
        generate_ir_pic = self.generator(ir_body_feats_g)


        # mm = cv2.imread('/data/hdd/guojunjie/pp-output-groupx3-missing-v3-m3fdmissing/original_pic/test.png')
        # cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v4-m3fdmissing/generate_pic/gt_vis.png',
        #             numpy.floor(np.transpose((numpy.array(self.inputs['vis_image'][0, :, :, :]) * 255), (1, 2, 0))))
        # cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v4-m3fdmissing/generate_pic/g_vis.png',
        #             numpy.floor(np.transpose((numpy.array(generate_vis_pic[0, :, :, :]) * 255), (1, 2, 0))))
        #
        # cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v4-m3fdmissing/generate_pic/gt_ir.png',
        #             numpy.floor(np.transpose((numpy.array(self.inputs['ir_image'][0, :, :, :]) * 255), (1, 2, 0))))
        # cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v4-m3fdmissing/generate_pic/g_ir.png',numpy.floor(np.transpose((numpy.array(generate_ir_pic[0,:,:,:])*255),(1,2,0))))
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        out_transformer = self.transformer(None,vis_body_feats, ir_body_feats,vis_body_feats_g,ir_body_feats_g, pad_mask, self.inputs)

        out_transformer = out_transformer + (generate_vis_pic,generate_ir_pic,)

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
            preds = (preds[0][:,600:900,:],preds[1][:,600:900,:],preds[2])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
