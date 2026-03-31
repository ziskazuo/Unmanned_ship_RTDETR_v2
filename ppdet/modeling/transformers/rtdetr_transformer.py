# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import time
import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from ..heads.detr_head import MLP
from .deformable_transformer import MSDeformableAttention,MSDeformableAttention_key_aware,MSDeformableAttention_Missing,MSDeformableAttention_Missing_3samplinglayer,MSDeformableAttention_RadarCamera
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_,
                           bias_init_with_prob, vector_)
from .utils import (_get_clones, get_sine_pos_embed,
                    get_contrastive_denoising_training_group, inverse_sigmoid)
try:
    from .utils import get_contrastive_denoising_training_group_rotated
except ImportError:
    def get_contrastive_denoising_training_group_rotated(*args, **kwargs):
        raise NotImplementedError(
            "get_contrastive_denoising_training_group_rotated is required for rotated RT-DETR variants.")
import numpy as np
import cv2
logger = logging.getLogger(__name__)
__all__ = ['RTDETRTransformer','Multi_RTDETRTransformer_V2_3LEVEL','Multi_RTDETRTransformer','Multi_RTDETRTransformer_V3','Multi_RTDETRTransformer_V4',
           'Multi_Group_RTDETRTransformer_V3',
            'Multi_Groupx4_RTDETRTransformer_V3','Multi_RTDETRTransformer_V3_BA','Multi_RTDETRTransformer_V7','Multi_RTDETRTransformer_split',
           'Multi_RTDETRTransformerv2_6lselect_3LEVEL', 'Multi_RTDETRTransformer_V3_RANK', 'Multi_Groupx4_RTDETRTransformer_V3_RANK',
           'Multi_Groupx3_RTDETRTransformer_V3_Missing','Multi_Groupx5_RTDETRTransformer_V3_Missing',
           'Multi_Groupx3_RTDETRTransformer_Missing','Multi_Groupx3_RTDETRTransformer_Missing_V3',
           'Multi_RTDETRTransformer_RadarCamera', 'Multi_RTDETRTransformer_RadarCamera_Rotate']


class PPMSDeformableAttention(MSDeformableAttention):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # ## sample location visulize
        #
        #   ##plot location point
        # bs = len(gt_meta['im_id'])
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_location_point = np.array(sampling_locations) * real_hw
        # real_weights = np.array(attention_weights)
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        # min_score = 0.0
        # max_score = 0.15
        # yellow_color = (0, 255, 255)
        # red_color = (0, 0, 255)
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_score[xx][ii] > 0:
        #             #if topk_ind_mask[xx][ii] == 1:
        #             for mm in range(8):
        #                 for zz in range(2):
        #                     for pp in range(4):
        #                         if real_weights[xx][ii][mm][zz][pp] > 0.12:
        #                             cv2.circle(vis_imgs[xx],(round(real_location_point[xx][ii][mm][zz][pp][0]),round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius-2,color_r,-1)
        #                         elif real_weights[xx][ii][mm][zz][pp] > 0.08:
        #                             cv2.circle(vis_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz][pp][0]),round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius-3, color_g, -1)
        #
        #                         #color_value = int(255*(real_weights[xx][ii][mm][zz][pp]-min_score)/(max_score - min_score))
        #                         # color = tuple(int((1 - real_weights[xx][ii][mm][zz][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius-2,color,-1)
        #
        #
        #                         # else:
        #                         #     cv2.circle(vis_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #
        #
        #                         if real_weights[xx][ii][mm][zz + 2][pp] > 0.12:
        #                             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                                                       round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                                        radius - 2, color_r, -1)
        #                         elif real_weights[xx][ii][mm][zz+2][pp] > 0.08:
        #                             cv2.circle(ir_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                                        radius - 3, color_g, -1)
        #
        #                         # color_value2 = int(
        #                         #     255 * (real_weights[xx][ii][mm][zz + 2][pp] - min_score) / (max_score - min_score))
        #                         # color2 = tuple(int((1 - real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #            radius - 2, color2, -1)
        #
        #
        #                         # else:
        #                         #     cv2.circle(ir_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #             #####################no no no
        #             #elif topk_ind_mask[xx][ii] == 2:
        #                 #for mm in range(8):
        #                     # for zz in range(2):
        #                     #     for pp in range(4):
        #                     #         if real_weights[xx][ii][mm][zz+2][pp] > 0.1:
        #                     #             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                     #                                       round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                     #                        radius - 1, color_r, -1)
        #                     #         elif real_weights[xx][ii][mm][zz+2][pp] > 0.05:
        #                     #             cv2.circle(ir_imgs[xx],
        #                     #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                     #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                     #                        radius - 3, color_g, -1)
        #                     #         else:
        #                     #             cv2.circle(ir_imgs[xx],
        #                     #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                     #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                     #                        radius - 3, color_b, -1)
        #
        # for ii in range(bs):
        #     # heatmap_image1 = cv2.applyColorMap(vis_imgs[ii],cv2.COLORMAP_JET)
        #     # heatmap_image2 = cv2.applyColorMap(ir_imgs[ii],cv2.COLORMAP_JET)
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-oldm3fd-nobug/sample_location_0.12,0.08/'+'layer'+str(gt_meta['layer']+1)+'/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-oldm3fd-nobug/sample_location_0.12,0.08/'+'layer'+str(gt_meta['layer']+1)+'/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])

        # ## sample location of different semantic level visulize
        #
        # ##plot location point
        # bs = len(gt_meta['im_id'])
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h, w, _ = vis_imgs[0].shape
        # real_hw = [w, h]
        # real_hw = np.array(real_hw)
        # real_location_point = np.array(sampling_locations) * real_hw
        # real_weights = np.array(attention_weights)
        # radius = 4
        # color_r = (0, 0, 255)
        # color_b = (230, 216, 173)
        # color_g = (152, 251, 152)
        # min_score = 0.0
        # max_score = 0.15
        # yellow_color = (0, 255, 255)
        # red_color = (0, 0, 255)
        #
        # low_blue = (230, 216, 173)
        # low_green = (152, 251, 152)
        # low_purple = (186,85,211)
        # low_red = (128,128,240)
        # blue = (255,0,0)
        # green = (0,255,0)
        # purple = (128,0,128)
        # red = (0,0,255)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_score[xx][ii] > 0:
        #             # if topk_ind_mask[xx][ii] == 1:
        #             for mm in range(8):
        #                 for zz in range(3):
        #                     for pp in range(4):
        #                         if real_weights[xx][ii][mm][zz][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                                       round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(vis_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value = int(255*(real_weights[xx][ii][mm][zz][pp]-min_score)/(max_score - min_score))
        #                         # color = tuple(int((1 - real_weights[xx][ii][mm][zz][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius-2,color,-1)
        #
        #                         # else:
        #                         #     cv2.circle(vis_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #
        #                         if real_weights[xx][ii][mm][zz + 2][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz + 2][pp][0]),
        #                                                      round(real_location_point[xx][ii][mm][zz + 2][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz + 2][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(ir_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz + 2][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz + 2][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value2 = int(
        #                         #     255 * (real_weights[xx][ii][mm][zz + 2][pp] - min_score) / (max_score - min_score))
        #                         # color2 = tuple(int((1 - real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #            radius - 2, color2, -1)
        #
        #                         # else:
        #                         #     cv2.circle(ir_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #             #####################no no no
        #             # elif topk_ind_mask[xx][ii] == 2:
        #             # for mm in range(8):
        #             # for zz in range(2):
        #             #     for pp in range(4):
        #             #         if real_weights[xx][ii][mm][zz+2][pp] > 0.1:
        #             #             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                                       round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 1, color_r, -1)
        #             #         elif real_weights[xx][ii][mm][zz+2][pp] > 0.05:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_g, -1)
        #             #         else:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_b, -1)
        #
        # for ii in range(bs):
        #     # heatmap_image1 = cv2.applyColorMap(vis_imgs[ii],cv2.COLORMAP_JET)
        #     # heatmap_image2 = cv2.applyColorMap(ir_imgs[ii],cv2.COLORMAP_JET)
        #     cv2.imwrite(
        #         '/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-oldm3fd-nobug/sample_location_0.1,0.03_sematic/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0] + '_vis.png', vis_imgs[ii])
        #     cv2.imwrite(
        #         '/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-oldm3fd-nobug/sample_location_0.1,0.03_sematic/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0] + '_ir.png', ir_imgs[ii])





        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:

            value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
            value_level_start_index = paddle.to_tensor(value_level_start_index)
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output



class PPMSDeformableAttention_RadarCamera(MSDeformableAttention_RadarCamera):

    def angle2rad(self, angle):
        """将角度转换为弧度"""
        return np.deg2rad(angle)

    def lidar_to_camera_transform(self, point_data_matrix, offset_x, offset_y, offset_z, pitch_x, yaw_y, roll_z,
                                  rotation_lidar_2_camera, camera_intrinsic):
        """将激光雷达点云数据转换为相机坐标，并返回像素坐标"""

        # 坐标系方向变化

        # 角度转弧度
        pitch_x = self.angle2rad(pitch_x)
        yaw_y = self.angle2rad(yaw_y)
        roll_z = self.angle2rad(roll_z)

        # 旋转矩阵
        rotation_x = paddle.to_tensor([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_x), np.sin(pitch_x)],
            [0.0, -np.sin(pitch_x), np.cos(pitch_x)]
        ])

        rotation_y = paddle.to_tensor([
            [np.cos(yaw_y), 0.0, -np.sin(yaw_y)],
            [0.0, 1.0, 0.0],
            [np.sin(yaw_y), 0.0, np.cos(yaw_y)]
        ])

        rotation_z = paddle.to_tensor([
            [np.cos(roll_z), np.sin(roll_z), 0.0],
            [-np.sin(roll_z), np.cos(roll_z), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # 平移矩阵 (3x1)
        translation_matrix = paddle.to_tensor([[offset_x], [offset_y], [offset_z]]).astype('float32')

        # 激光雷达点矩阵与旋转矩阵的转换 (将点从激光雷达坐标系转换到相机坐标系)
        result = paddle.to_tensor(np.ones((1, point_data_matrix.shape[1])))  # 1xN

        # 激光雷达到相机的坐标转换（包括旋转和平移）

        # lidar_2_camera_data = camera_intrinsic @ (
        #             (rotation_x @ rotation_y @ rotation_z) @ (rotation_lidar_2_camera @ point_data_matrix) + (
        #                 translation_matrix @ result))

        bs = point_data_matrix.shape[0]
        num = point_data_matrix.shape[2]
        R_total = rotation_x @ rotation_y @ rotation_z @ rotation_lidar_2_camera  # [3,3]
        R_total = R_total.unsqueeze(0).tile([bs, 1, 1])  # [bs,3,3]
        lidar_2_camera_data = camera_intrinsic.unsqueeze(0).tile([bs, 1, 1]) @ (
                R_total @ point_data_matrix + translation_matrix.unsqueeze(0).tile([bs, 1, num])
        )

        # 获取像素坐标
        #pixel_data = []
        pixel_cols = float(2048)  # 假设的图像宽度
        pixel_rows = float(2048)  # 假设的图像高度

        # for i in range(lidar_2_camera_data.shape[1]):
        # u = int(lidar_2_camera_data[0, i] / lidar_2_camera_data[2, i])
        # v = int(lidar_2_camera_data[1, i] / lidar_2_camera_data[2, i])

        # for i in range(lidar_2_camera_data.shape[3]):
        #     u = int(lidar_2_camera_data[:, :, 0, i] / lidar_2_camera_data[:, :, 2, i])
        #     v = int(lidar_2_camera_data[:, :, 1, i] / lidar_2_camera_data[:, :, 2, i])
        #
        #     if 0 <= u <= pixel_cols and 0 <= v <= pixel_rows:
        #         pixel_data.append([u, v])

        # 假设 lidar_2_camera_data 为 [1, 3, 300]
        # 取出 u, v
        u = lidar_2_camera_data[:, 0, :] / (lidar_2_camera_data[:, 2, :] + 1e-6)
        v = lidar_2_camera_data[:, 1, :] / (lidar_2_camera_data[:, 2, :] + 1e-6)

        # pixel_cols / rows 确保 float32 类型
        pixel_cols = paddle.to_tensor(pixel_cols, dtype='float32')
        pixel_rows = paddle.to_tensor(pixel_rows, dtype='float32')

        # 计算 mask，保持为 bool 类型
        mask = (u >= 0) & (u <= pixel_cols) & (v >= 0) & (v <= pixel_rows)

        # 堆叠 u 和 v，shape: [1, 300, 2]
        pixel_data = paddle.stack([u, v], axis=2)

        # 为 mask 添加维度 -> [1, 300, 1] 再扩展成 [1, 300, 2]
        mask = paddle.unsqueeze(mask, axis=2).expand([-1, -1, 2])

        # 替换非法值为 -1（保持 dtype 与 pixel_data 一致）
        replace_value = paddle.full_like(pixel_data, -1.0, dtype=pixel_data.dtype)

        #normalize
        pixel_data[:,:,0] = pixel_data[:,:,0] / 2048
        pixel_data[:, :, 1] = pixel_data[:, :, 1] / 2048

        # ✅ 核心点：三个输入 dtype 一致：where(bool, float32, float32)
        pixel_data = paddle.where(mask, pixel_data, replace_value)

        return pixel_data, mask


    def forward(self,
                gt_meta,
                query,
                reference_points,
                radar_value,
                camera_value,
                radar_value_spatial_shapes,
                radar_value_level_start_index,
                camera_value_spatial_shapes,
                camera_value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None):
        if not hasattr(self, '_debug_gt_meta_printed'):
            # try:
            #     print('DEBUG gt_meta keys:', list(gt_meta.keys()))
            # except Exception as exc:
            #     print('DEBUG gt_meta inspect failed:', exc)
            self._debug_gt_meta_printed = True
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v_radar = radar_value.shape[1]
        Len_v_camera = camera_value.shape[1]

        radar_value = self.value_proj(radar_value)
        camera_value = self.value_proj(camera_value)
        if value_mask is not None:
            value_mask = value_mask.astype(radar_value.dtype).unsqueeze(-1)
            radar_value *= value_mask
        radar_value = radar_value.reshape([bs, Len_v_radar, self.num_heads, self.head_dim])
        camera_value = camera_value.reshape([int(bs*6), Len_v_camera, self.num_heads, self.head_dim])

        # radar&camera sampling
        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        # radar sampling locations
        radar_sampling_locations = (
            reference_points[:, :, None, :, None, :2] + sampling_offsets[:,:,:,:3,:,:] /
            self.num_points * reference_points[:, :, None, :, None, 2:] *
            0.5)
        _,_,n_head,_,n_num,_ = radar_sampling_locations.shape
        # camera sampling locations
        camera_reference_points = reference_points[:, :, 0:1, :2]
        im_shape = gt_meta['im_shape']
        if len(im_shape.shape) == 2:
            im_shape = paddle.unsqueeze(im_shape, axis=1)
            im_shape = paddle.unsqueeze(im_shape, axis=1)
        camera_reference_points_real = camera_reference_points * im_shape
        center_y = int(gt_meta['im_shape'][0,0]) // 2
        center_x = int(gt_meta['im_shape'][0,1]) // 2

        # 单位距离转换 ppi图像半径为1024, 探测范围为200000mm, 所以单位距离为 (2000000/1024) mm
        unit = (14424 / 66) / 1000  # 单位m
        camera_reference_points_real[:,:,:,0] = (center_x - camera_reference_points_real[:,:,:,0]) * unit
        camera_reference_points_real[:, :, :, 1] = (center_y - camera_reference_points_real[:, :, :, 1]) * unit

        #add_tensor = paddle.full([1, 300, 1, 1], 5.0, dtype=camera_reference_points_real.dtype)
        cam_len = camera_reference_points_real.shape[1]
        batch = camera_reference_points_real.shape[0]
        add_tensor = paddle.full([batch, cam_len, 1, 1], 5.0,
                                 dtype=camera_reference_points_real.dtype)        
        camera_reference_points_real = paddle.concat([camera_reference_points_real, add_tensor], axis=-1)
        camera_reference_points_real = paddle.squeeze(camera_reference_points_real, axis=2)
        camera_reference_points_real = paddle.transpose(camera_reference_points_real, perm=[0, 2, 1])

        # 外参
        offset_x, offset_y, offset_z = 0, -1, 0
        pitch_x, yaw_y, roll_z = 0, [0, 60, 120, 180, 240, 300], 0

        # 示例的激光雷达到相机的旋转矩阵
        rotation_lidar_2_camera = paddle.to_tensor([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ])  # 假设的 3x3 单位矩阵

        # 相机的内参矩阵 (假设的内参矩阵)
        camera_intrinsic = paddle.to_tensor([
            [1773.62, 0, 1024],  # fx, 0, cx
            [0, 1773.62, 1024],  # 0, fy, cy
            [0, 0, 1]  # 0, 0, 1
        ])

        # 基于真实AirSim参数的相机配置
        # 从camera_calibration.py提取的真实参数
        real_camera_intrinsic = paddle.to_tensor([
            [500.0, 0, 500.0],  # 真实的焦距和主点
            [0, 500.0, 500.0],
            [0, 0, 1]
        ], dtype='float32')
        
        # 真实的相机外参 (4个相机: Front, Right, Back, Left)
        real_camera_poses = [
            0,    # Front (cam_1) - yaw = 0度
            90,   # Right (cam_2) - yaw = 90度
            180,  # Back (cam_3) - yaw = 180度  
            -90   # Left (cam_4) - yaw = -90度
        ]

        offset_normalizer = paddle.to_tensor(camera_value_spatial_shapes)
        offset_normalizer = offset_normalizer.flip([1]).reshape(
            [1, 1, 1, self.num_levels // 2, 1, 2])

        camera_reference_points_list = []
        camera_mask_list = []
        camera_sampling_locations_list = []
        
        # 根据实际相机数量处理，这里camera_value.shape[0]=6表示6个相机视角
        num_total_cameras = int(camera_value.shape[0] // bs)
        
        # 但我们只有4个真实相机的参数，所以对于额外的2个，使用默认处理
        for i in range(num_total_cameras):
            if i < len(real_camera_poses):
                # 使用真实参数进行投影 (前4个相机)
                yaw_angle = real_camera_poses[i]
                
                pixel_data, mask = self.lidar_to_camera_transform(
                    camera_reference_points_real, 
                    offset_x, offset_y, offset_z, 
                    pitch_x, yaw_angle, roll_z,
                    rotation_lidar_2_camera, 
                    real_camera_intrinsic
                )
            else:
                # 对于额外的2个相机 (TopView, Bottom)，使用原来的方法
                # TopView: yaw=0, pitch=-90; Bottom: yaw=0, pitch=90
                if i == 4:  # TopView
                    extra_yaw = 0
                    extra_pitch = -90
                else:  # Bottom (i == 5)
                    extra_yaw = 0  
                    extra_pitch = 90
                    
                pixel_data, mask = self.lidar_to_camera_transform(
                    camera_reference_points_real, 
                    offset_x, offset_y, offset_z, 
                    extra_pitch, extra_yaw, roll_z,
                    rotation_lidar_2_camera, 
                    real_camera_intrinsic
                )
            
            camera_reference_points_list.append(paddle.unsqueeze(pixel_data, axis=2))
            camera_mask_list.append(paddle.unsqueeze(mask, axis=2))
            camera_mask_list[i] = camera_mask_list[i].unsqueeze(2).unsqueeze(3).expand([bs, Len_q, n_head, self.num_levels // 2, n_num, 2])
            
            camera_sampling_locations = camera_reference_points_list[i].expand([-1, -1, self.num_levels // 2, -1]).reshape([
                bs, Len_q, 1, self.num_levels // 2, 1, 2
            ]) + sampling_offsets[:,:,:,3:,:,:] / offset_normalizer
            camera_sampling_locations_list.append(camera_sampling_locations)

        radar_value_spatial_shapes = paddle.to_tensor(radar_value_spatial_shapes)
        radar_value_level_start_index = paddle.to_tensor(radar_value_level_start_index)
        camera_value_spatial_shapes = paddle.to_tensor(camera_value_spatial_shapes)
        camera_value_level_start_index = paddle.to_tensor(camera_value_level_start_index)
        output = self.ms_deformable_attn_core(
            radar_value, camera_value, radar_value_spatial_shapes, radar_value_level_start_index,
            camera_value_spatial_shapes, camera_value_level_start_index,
            radar_sampling_locations, camera_sampling_locations_list, camera_mask_list, attention_weights)
        output = self.output_proj(output)

        return output

class PPMSDeformableAttention_Rotate(MSDeformableAttention):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                angle,
                value,
                value_spatial_shapes,
                value_level_start_index,
                angle_max,
                half_pi_bin,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                mask_vis = None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间

            rotate_part1 = paddle.concat([paddle.cos(angle), paddle.sin(angle)], axis=-1)
            rotate_part2 = paddle.concat([-paddle.sin(angle), paddle.cos(angle)], axis=-1)
            rotate_matrix = paddle.stack([rotate_part1, rotate_part2], axis=-2)

            rotate_matrix = paddle.broadcast_to(rotate_matrix[:, :, None, None],
                                                [bs, Len_q, self.num_heads, self.num_levels, 2, 2])

            sampling_locations = reference_points[:, :, None, :, None, :2] + paddle.matmul(
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5,
                rotate_matrix)

        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:

            value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
            value_level_start_index = paddle.to_tensor(value_level_start_index)
            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output
class PPMSDeformableAttention_RadarCamera_Rotate(MSDeformableAttention_RadarCamera):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1,
                 height_bins=32,
                 height_min=-20.0,
                 height_max=20.0,
                 visualize=True,
                 visualize_dir='attn_visualization2',
                 visualize_interval=50,#每前向20次可视化一次
                 visualize_max_samples=1000,#最多可视化1000个样本
                 visualize_num_queries=20,#可视化前20个查询
                 visualize_head_index=0,#如果visualize_head_indices设置为None，这个默认第0个head
                 visualize_head_indices=[0,1,2]):
        super(PPMSDeformableAttention_RadarCamera_Rotate, self).__init__(
            embed_dim, num_heads, num_levels, num_points, lr_mult)
        self.register_buffer(
            'fixed_height', paddle.to_tensor([-10.0], dtype='float32'))
        self.visualize = bool(visualize)
        self.visualize_dir = str(visualize_dir) if visualize_dir else 'output/attn_vis'
        self.visualize_interval = max(1, int(visualize_interval)) if visualize_interval else 1
        self.visualize_max_samples = (None if visualize_max_samples is None else
                                      max(1, int(visualize_max_samples)))
        self.visualize_num_queries = max(1, int(visualize_num_queries)) if visualize_num_queries else 1
        if visualize_head_indices is not None:
            if isinstance(visualize_head_indices, (list, tuple)):
                self.visualize_heads = [int(h) for h in visualize_head_indices]
            else:
                self.visualize_heads = [int(visualize_head_indices)]
        else:
            self.visualize_heads = [int(visualize_head_index) if visualize_head_index is not None else 0]
        self._visualize_call_count = 0
        self._visualize_saved = 0
        self._visualize_warned = False

    @staticmethod
    def _to_tensor(value, dtype='float32'):
        if isinstance(value, paddle.Tensor):
            return value.astype(dtype)
        return paddle.to_tensor(value, dtype=dtype)

    @staticmethod
    def angle2rad(angle):
        return np.deg2rad(angle)

    @staticmethod
    def _ensure_dir(path):
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _tensor_to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, paddle.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _prepare_image_for_vis(self, image):
        if image is None:
            return None
        img = self._tensor_to_numpy(image)
        if img is None:
            return None
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-6:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def _extract_radar_image(self, gt_meta):
        if not isinstance(gt_meta, dict):
            return None
        radar = gt_meta.get('radar_image')
        radar_np = self._tensor_to_numpy(radar) if radar is not None else None
        if radar_np is None:
            im_shape = self._tensor_to_numpy(gt_meta.get('im_shape'))
            if im_shape is not None:
                height = int(im_shape[0][0])
                width = int(im_shape[0][1])
                return np.zeros((height, width, 3), dtype=np.uint8)
            return None
        if radar_np.ndim == 4:
            radar_np = radar_np[0]
        return self._prepare_image_for_vis(radar_np)

    def _extract_camera_images(self, gt_meta):
        images = []
        if not isinstance(gt_meta, dict):
            return images
        camera_value = gt_meta.get('camera_image')
        if camera_value is not None:
            if isinstance(camera_value, paddle.Tensor):
                cam_np = self._tensor_to_numpy(camera_value)
                if cam_np is None:
                    cam_np = None
                else:
                    if cam_np.ndim == 5:
                        cam_np = cam_np[0]
                    if cam_np.ndim == 4:
                        for idx in range(cam_np.shape[0]):
                            images.append(self._prepare_image_for_vis(cam_np[idx]))
                    elif cam_np.ndim == 3:
                        images.append(self._prepare_image_for_vis(cam_np))
            elif isinstance(camera_value, (list, tuple)):
                for cam in camera_value:
                    cam_np = self._tensor_to_numpy(cam)
                    if cam_np is None:
                        continue
                    if cam_np.ndim == 4 and cam_np.shape[0] == 1:
                        cam_np = cam_np[0]
                    images.append(self._prepare_image_for_vis(cam_np))
        if not images:
            size_info = self._tensor_to_numpy(gt_meta.get('camera_img_size'))
            if size_info is not None:
                if size_info.ndim == 3:
                    size_info = size_info[0]
                for cam_idx in range(size_info.shape[0]):
                    width = int(size_info[cam_idx][0])
                    height = int(size_info[cam_idx][1])
                    images.append(np.zeros((height, width, 3), dtype=np.uint8))
        return images

    def _draw_rotated_box(self, image, center, size, angle, color, thickness=1):
        if image is None:
            return
        w, h = size
        if w <= 0 or h <= 0:
            return
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        half_w, half_h = w / 2.0, h / 2.0
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rotation.T + center
        pts = np.round(rotated).astype(np.int32)
        cv2.polylines(image, [pts], True, color, thickness, lineType=cv2.LINE_AA)

    def _maybe_visualize(self,
                         gt_meta,
                         reference_points,
                         radar_sampling_locations,
                         camera_reference_points_list,
                         camera_sampling_locations_list,
                         camera_mask_list,
                         radar_value_spatial_shapes,
                         camera_value_spatial_shapes,
                         angle,
                         attention_weights):
        if not self.visualize or not isinstance(gt_meta, dict):
            return
        self._visualize_call_count += 1
        if self._visualize_call_count % self.visualize_interval != 0:
            return
        if self.visualize_max_samples is not None and \
                self._visualize_saved >= self.visualize_max_samples:
            return
        try:
            self._create_visualizations(
                gt_meta,
                reference_points,
                radar_sampling_locations,
                camera_reference_points_list,
                camera_sampling_locations_list,
                camera_mask_list,
                radar_value_spatial_shapes,
                camera_value_spatial_shapes,
                angle,
                attention_weights)
            self._visualize_saved += 1
        except Exception as exc:
            if not self._visualize_warned:
                logger.warning("Sampling visualization failed: %s", exc)
                self._visualize_warned = True

    def _create_visualizations(self,
                               gt_meta,
                               reference_points,
                               radar_sampling_locations,
                               camera_reference_points_list,
                               camera_sampling_locations_list,
                               camera_mask_list,
                               radar_value_spatial_shapes,
                               camera_value_spatial_shapes,
                               angle,
                               attention_weights):
        radar_img = self._extract_radar_image(gt_meta)
        camera_imgs = self._extract_camera_images(gt_meta)
        ref_np = self._tensor_to_numpy(reference_points)
        if ref_np is None or ref_np.shape[0] == 0:
            return
        cam_ref_np = [
            self._tensor_to_numpy(val) for val in camera_reference_points_list
        ]
        bs = ref_np.shape[0]
        if bs == 0:
            return
        query_count = ref_np.shape[1]
        selected_queries = range(min(self.visualize_num_queries, query_count))
        palette = [
            (0, 0, 255),
            (0, 128, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
            (128, 0, 255),
            (255, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (0, 128, 128),
            (0, 128, 0),
            (128, 0, 0),
            (128, 0, 128),
            (255, 128, 128),
            (255, 0, 128),
            (128, 255, 0),
            (0, 255, 128),
            (128, 128, 255),
        ]

        def _scalar_from_meta(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                value = value[0]
            if isinstance(value, paddle.Tensor):
                value = self._tensor_to_numpy(value)
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                value = value.reshape(-1)[0]
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        frame_id = None
        cam_indices = None
        if isinstance(gt_meta, dict):
            frame_id = _scalar_from_meta(gt_meta.get('frame', None))
            cam_indices = gt_meta.get('camera_names', None)
            if frame_id is None:
                frame_id = _scalar_from_meta(gt_meta.get('im_id', None))
        if frame_id is None:
            frame_id = self._visualize_saved
        frame_str = f"{int(frame_id):04d}"

        camera_label_map = {
            0: 'back',
            1: 'front',
            2: 'left',
            3: 'right'
        }
        if isinstance(cam_indices, (np.ndarray, paddle.Tensor)):
            cam_indices = self._tensor_to_numpy(cam_indices)
        if isinstance(cam_indices, (list, tuple)):
            camera_labels = [
                camera_label_map.get(int(idx), f"cam{cam_idx}")
                for cam_idx, idx in enumerate(cam_indices)
            ]
        else:
            camera_labels = [
                camera_label_map.get(cam_idx, f"cam{cam_idx}")
                for cam_idx in range(len(camera_imgs))
            ]

        frame_dir = os.path.join(self.visualize_dir, frame_str)
        self._ensure_dir(frame_dir)
        radar_path = os.path.join(frame_dir, f"{frame_str}_radar.png")

        if radar_img is not None:
            radar_vis = radar_img.copy()
            img_h, img_w = radar_vis.shape[:2]
            for qi in selected_queries:
                color = palette[qi % len(palette)]
                if ref_np.ndim < 3 or ref_np.shape[2] == 0:
                    continue
                center = ref_np[0, qi, 0, :2] * np.array([img_w, img_h])
                if np.isnan(center).any():
                    continue
                center_int = tuple(np.round(center).astype(int))
                if 0 <= center_int[0] < img_w and 0 <= center_int[1] < img_h:
                    cv2.circle(radar_vis, center_int, 3, color, -1, lineType=cv2.LINE_AA)
            cv2.imwrite(radar_path, cv2.cvtColor(radar_vis, cv2.COLOR_RGB2BGR))

        for cam_idx, cam_img in enumerate(camera_imgs):
            if cam_idx >= len(cam_ref_np) or cam_ref_np[cam_idx] is None:
                continue
            camera_vis = cam_img.copy()
            cam_h, cam_w = camera_vis.shape[:2]
            cam_refs = cam_ref_np[cam_idx]
            if cam_refs is None or cam_refs.size == 0:
                continue
            cam_label = camera_labels[cam_idx] if cam_idx < len(camera_labels) else f"cam{cam_idx}"
            cam_filename = f"{frame_str}_{cam_label}.png"
            for qi in selected_queries:
                color = palette[qi % len(palette)]
                if cam_refs.ndim >= 4:
                    ref_candidates = cam_refs[0, qi]
                elif cam_refs.ndim == 3:
                    ref_candidates = cam_refs[0, qi:qi + 1]
                elif cam_refs.ndim == 2:
                    ref_candidates = cam_refs[qi:qi + 1]
                else:
                    continue
                ref_candidates = np.reshape(ref_candidates, (-1, 2))
                if ref_candidates.size == 0:
                    continue
                ref_pt = ref_candidates[0]
                if np.isnan(ref_pt).any():
                    continue
                if (ref_pt < 0).any() or ref_pt[0] > 1.0 or ref_pt[1] > 1.0:
                    continue
                pt = ref_pt * np.array([cam_w, cam_h])
                pt_int = tuple(np.round(pt).astype(int))
                if 0 <= pt_int[0] < cam_w and 0 <= pt_int[1] < cam_h:
                    cv2.circle(camera_vis, pt_int, 3, color, -1, lineType=cv2.LINE_AA)
            cam_path = os.path.join(frame_dir, cam_filename)
            cv2.imwrite(cam_path, cv2.cvtColor(camera_vis, cv2.COLOR_RGB2BGR))

    def lidar_to_camera_transform(self, point_data_matrix, offset_x, offset_y, offset_z, pitch_x, yaw_y, roll_z,
                                  rotation_lidar_2_camera, camera_intrinsic):
        pitch_x = self.angle2rad(pitch_x)
        yaw_y = self.angle2rad(yaw_y)
        roll_z = self.angle2rad(roll_z)

        rotation_x = paddle.to_tensor([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_x), np.sin(pitch_x)],
            [0.0, -np.sin(pitch_x), np.cos(pitch_x)]
        ])

        rotation_y = paddle.to_tensor([
            [np.cos(yaw_y), 0.0, -np.sin(yaw_y)],
            [0.0, 1.0, 0.0],
            [np.sin(yaw_y), 0.0, np.cos(yaw_y)]
        ])

        rotation_z = paddle.to_tensor([
            [np.cos(roll_z), np.sin(roll_z), 0.0],
            [-np.sin(roll_z), np.cos(roll_z), 0.0],
            [0.0, 0.0, 1.0]
        ])

        translation_matrix = paddle.to_tensor([[offset_x], [offset_y], [offset_z]]).astype('float32')
        result = paddle.to_tensor(np.ones((1, point_data_matrix.shape[1])))

        bs = point_data_matrix.shape[0]
        num = point_data_matrix.shape[2]
        R_total = rotation_x @ rotation_y @ rotation_z @ rotation_lidar_2_camera
        R_total = R_total.unsqueeze(0).tile([bs, 1, 1])
        lidar_2_camera_data = camera_intrinsic.unsqueeze(0).tile([bs, 1, 1]) @ (
            R_total @ point_data_matrix + translation_matrix.unsqueeze(0).tile([bs, 1, num]))

        pixel_cols = float(2048)
        pixel_rows = float(2048)
        u = lidar_2_camera_data[:, 0, :] / (lidar_2_camera_data[:, 2, :] + 1e-6)
        v = lidar_2_camera_data[:, 1, :] / (lidar_2_camera_data[:, 2, :] + 1e-6)

        pixel_cols = paddle.to_tensor(pixel_cols, dtype='float32')
        pixel_rows = paddle.to_tensor(pixel_rows, dtype='float32')

        mask = (u >= 0) & (u <= pixel_cols) & (v >= 0) & (v <= pixel_rows)
        pixel_data = paddle.stack([u, v], axis=2)
        mask = paddle.unsqueeze(mask, axis=2).expand([-1, -1, 2])
        replace_value = paddle.full_like(pixel_data, -1.0, dtype=pixel_data.dtype)
        pixel_data[:, :, 0] = pixel_data[:, :, 0] / 2048
        pixel_data[:, :, 1] = pixel_data[:, :, 1] / 2048
        pixel_data = paddle.where(mask, pixel_data, replace_value)
        return pixel_data, mask

    def forward(self,
                gt_meta,
                query,
                reference_points,
                angle,
                radar_value,
                camera_value,
                radar_value_spatial_shapes,
                radar_value_level_start_index,
                camera_value_spatial_shapes,
                camera_value_level_start_index,
                angle_max,
                half_pi_bin,
                value_mask=None,
                topk_ind_mask=None,
                topk_score=None,
                sort_index=None,
                mask_vis=None):
        bs, Len_q = query.shape[:2]
        Len_v_radar = radar_value.shape[1]
        Len_v_camera = camera_value.shape[1]

        radar_value = self.value_proj(radar_value)
        camera_value = self.value_proj(camera_value)
        if value_mask is not None:
            value_mask = value_mask.astype(radar_value.dtype).unsqueeze(-1)
            radar_value *= value_mask
        radar_value = radar_value.reshape([bs, Len_v_radar, self.num_heads, self.head_dim])

        num_total_cameras = int(camera_value.shape[0] // bs)
        camera_value = camera_value.reshape(
            [int(bs * num_total_cameras), Len_v_camera, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        num_radar_levels = len(radar_value_spatial_shapes)
        radar_offsets = sampling_offsets[:, :, :, :num_radar_levels, :, :]

        if reference_points.shape[-1] == 4 and angle is not None:
            rotate_part1 = paddle.concat([paddle.cos(angle), paddle.sin(angle)], axis=-1)
            rotate_part2 = paddle.concat([-paddle.sin(angle), paddle.cos(angle)], axis=-1)
            rotate_matrix = paddle.stack([rotate_part1, rotate_part2], axis=-2)
            rotate_matrix = paddle.broadcast_to(
                rotate_matrix[:, :, None, None],
                [bs, Len_q, self.num_heads, num_radar_levels, 2, 2])
            radar_sampling_locations = reference_points[:, :, None, :num_radar_levels, None, :2] + paddle.matmul(
                radar_offsets / self.num_points *
                reference_points[:, :, None, :num_radar_levels, None, 2:] * 0.5,
                rotate_matrix)
        else:
            radar_sampling_locations = (
                reference_points[:, :, None, :num_radar_levels, None, :2] +
                radar_offsets / self.num_points *
                reference_points[:, :, None, :num_radar_levels, None, 2:] * 0.5)

        _, _, n_head, _, n_num, _ = radar_sampling_locations.shape

        camera_reference_points_list = []
        camera_mask_list = []
        camera_sampling_locations_list = []

        use_calibration = isinstance(gt_meta, dict) and gt_meta.get('camera_intrinsics', None) is not None and gt_meta.get('lidar_T_ego', None) is not None

        if use_calibration:
            to_tensor = self._to_tensor
            im_shape = to_tensor(gt_meta['im_shape']).astype('float32')
            if im_shape.ndim == 1:
                im_shape = im_shape.unsqueeze(0)
            camera_intrinsics = to_tensor(gt_meta['camera_intrinsics']).astype('float32')
            if camera_intrinsics.ndim == 3:
                camera_intrinsics = camera_intrinsics.unsqueeze(0)
            camera_extrinsics = to_tensor(gt_meta['camera_extrinsics']).astype('float32')
            if camera_extrinsics.ndim == 3:
                camera_extrinsics = camera_extrinsics.unsqueeze(0)
            camera_img_size = to_tensor(gt_meta['camera_img_size']).astype('float32')
            if camera_img_size.ndim == 2:
                camera_img_size = camera_img_size.unsqueeze(0)

            if 'meters_per_pixel' in gt_meta and gt_meta['meters_per_pixel'] is not None:
                meters_per_pixel = to_tensor(gt_meta['meters_per_pixel']).astype('float32')
            else:
                lidar_range = to_tensor(gt_meta.get('lidar_range', paddle.ones([bs, 1]) * 1000.0)).astype('float32')
                ppi_res = to_tensor(gt_meta.get('ppi_res', im_shape[:, 1:2])).astype('float32')
                meters_per_pixel = lidar_range / paddle.maximum(ppi_res / 2.0, paddle.full_like(lidar_range, 1e-6))
            if meters_per_pixel.ndim == 1:
                meters_per_pixel = meters_per_pixel.unsqueeze(-1)
            meters_per_pixel = meters_per_pixel.astype('float32')

            if 'projection_plane_height' in gt_meta and \
                    gt_meta['projection_plane_height'] is not None:
                height_value = to_tensor(gt_meta['projection_plane_height']).astype(query.dtype)
            else:
                height_value = self.fixed_height.astype(query.dtype)

            if height_value.ndim == 0:
                height = height_value.reshape([1, 1, 1]).tile([bs, Len_q, 1])
            elif height_value.ndim == 1:
                if height_value.shape[0] == bs:
                    height = height_value.reshape([bs, 1, 1]).tile([1, Len_q, 1])
                else:
                    height = height_value.reshape([1, 1, 1]).tile([bs, Len_q, 1])
            else:
                height = height_value.reshape([bs, 1, 1]).tile([1, Len_q, 1])

            meters_scale = meters_per_pixel
            if meters_scale.ndim == 2:
                meters_scale = meters_scale.unsqueeze(1)

            center_ref = reference_points[:, :, 0, :2]
            im_height = im_shape[:, 0:1].unsqueeze(1)
            im_width = im_shape[:, 1:2].unsqueeze(1)
            pixel_u = center_ref[..., 0:1] * im_width
            pixel_v = center_ref[..., 1:2] * im_height
            center_u = im_width / 2.0
            center_v = im_height / 2.0
            delta_u = pixel_u - center_u  # 正向指向右侧
            delta_v = center_v - pixel_v  # 正向指向前方
            x_forward = delta_v * meters_scale
            y_right = delta_u * meters_scale

            lidar_points = paddle.concat([x_forward, y_right, height], axis=-1)
            ones = paddle.ones([bs, Len_q, 1], dtype=lidar_points.dtype)
            lidar_points_h = paddle.concat([lidar_points, ones], axis=-1)
            points_ego = lidar_points_h

            camera_offsets = sampling_offsets[:, :, :, num_radar_levels:, :, :]
            offset_normalizer = paddle.to_tensor(camera_value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels // 2, 1, 2])

            num_cams = camera_intrinsics.shape[1]
            for cam_idx in range(num_cams):
                cam_intr = camera_intrinsics[:, cam_idx, :, :]
                cam_ext = camera_extrinsics[:, cam_idx, :, :]
                cam_size = camera_img_size[:, cam_idx, :]
                if cam_idx == 4:
                    pixel_data = reference_points[:, :, 0, :2]
                    mask_bool = paddle.ones_like(pixel_data, dtype='bool')

                    camera_reference_points_list.append(pixel_data.unsqueeze(2))
                    mask_tensor = mask_bool.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                    mask_tensor = mask_tensor.tile([1, 1, n_head, self.num_levels // 2, n_num, 1])
                    camera_mask_list.append(mask_tensor.astype('bool'))

                    base = pixel_data.unsqueeze(2)
                    base = base.tile([1, 1, self.num_levels // 2, 1])
                    base = base.reshape([bs, Len_q, self.num_levels // 2, 1, 2])
                    base = base.unsqueeze(2)
                    camera_sampling_locations = base + camera_offsets / offset_normalizer
                    camera_sampling_locations_list.append(camera_sampling_locations)
                    continue

                points_cam = paddle.matmul(points_ego, paddle.transpose(cam_ext, [0, 2, 1]))
                X = points_cam[..., 0]
                Y = points_cam[..., 1]
                Z = points_cam[..., 2]
                Z_safe = paddle.where(Z > 1e-6, Z, paddle.ones_like(Z) * 1e-6)

                fx = cam_intr[:, 0, 0].reshape([bs, 1])
                fy = cam_intr[:, 1, 1].reshape([bs, 1])
                cx = cam_intr[:, 0, 2].reshape([bs, 1])
                cy = cam_intr[:, 1, 2].reshape([bs, 1])

                u = fx * (X / Z_safe) + cx
                v = fy * (Y / Z_safe) + cy

                width = cam_size[:, 0].reshape([bs, 1])
                height_img = cam_size[:, 1].reshape([bs, 1])
                width_safe = paddle.maximum(width, paddle.full_like(width, 1e-6))
                height_safe = paddle.maximum(height_img, paddle.full_like(height_img, 1e-6))

                u_norm = u / width_safe
                v_norm = v / height_safe

                valid = (Z > 1e-3) & (u >= 0.0) & (u <= width) & (v >= 0.0) & (v <= height_img)
                mask_bool = paddle.tile(valid.unsqueeze(-1), [1, 1, 2])

                pixel_data = paddle.stack([u_norm, v_norm], axis=-1)
                replace_value = paddle.full_like(pixel_data, -1.0)
                pixel_data = paddle.where(mask_bool, pixel_data, replace_value)

                camera_reference_points_list.append(pixel_data.unsqueeze(2))
                mask_tensor = mask_bool.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                mask_tensor = mask_tensor.tile([1, 1, n_head, self.num_levels // 2, n_num, 1])
                camera_mask_list.append(mask_tensor.astype('bool'))

                base = pixel_data.unsqueeze(2)
                base = base.tile([1, 1, self.num_levels // 2, 1])
                base = base.reshape([bs, Len_q, self.num_levels // 2, 1, 2])
                base = base.unsqueeze(2)
                camera_sampling_locations = base + camera_offsets / offset_normalizer
                camera_sampling_locations_list.append(camera_sampling_locations)
        else:
            print('无标定，开始假设参数')
            camera_reference_points = reference_points[:, :, 0:1, :2]
            im_shape = gt_meta['im_shape']
            if len(im_shape.shape) == 2:
                im_shape = paddle.unsqueeze(im_shape, axis=1)
                im_shape = paddle.unsqueeze(im_shape, axis=1)
            camera_reference_points_real = camera_reference_points * im_shape
            center_y = int(gt_meta['im_shape'][0, 0]) // 2
            center_x = int(gt_meta['im_shape'][0, 1]) // 2

            unit = (14424 / 66) / 1000
            u_pixels = camera_reference_points_real[:, :, :, 0]
            v_pixels = camera_reference_points_real[:, :, :, 1]
            x_forward = (center_y - v_pixels) * unit
            y_right = (u_pixels - center_x) * unit
            camera_reference_points_real = paddle.stack(
                [x_forward, y_right], axis=-1)

            cam_len = camera_reference_points_real.shape[1]
            batch = camera_reference_points_real.shape[0]
            add_tensor = paddle.full([batch, cam_len, 1, 1], 5.0,
                                     dtype=camera_reference_points_real.dtype)
            camera_reference_points_real = paddle.concat(
                [camera_reference_points_real, add_tensor], axis=-1)
            camera_reference_points_real = paddle.squeeze(camera_reference_points_real, axis=2)
            camera_reference_points_real = paddle.transpose(camera_reference_points_real, perm=[0, 2, 1])

            offset_x, offset_y, offset_z = 0, -1, 0
            pitch_x, yaw_y, roll_z = 0, [0, 60, 120, 180, 240, 300], 0

            rotation_lidar_2_camera = paddle.to_tensor([
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0]
            ])

            camera_intrinsic = paddle.to_tensor([
                [1773.62, 0, 1024],
                [0, 1773.62, 1024],
                [0, 0, 1]
            ])

            real_camera_intrinsic = paddle.to_tensor([
                [500.0, 0, 500.0],
                [0, 500.0, 500.0],
                [0, 0, 1]
            ], dtype='float32')

            real_camera_poses = [0, 90, 180, -90]

            offset_normalizer = paddle.to_tensor(camera_value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels // 2, 1, 2])

            for i in range(num_total_cameras):
                if i < len(real_camera_poses):
                    yaw_angle = real_camera_poses[i]
                    pixel_data, mask = self.lidar_to_camera_transform(
                        camera_reference_points_real,
                        offset_x, offset_y, offset_z,
                        pitch_x, yaw_angle, roll_z,
                        rotation_lidar_2_camera,
                        real_camera_intrinsic)
                else:
                    if i == 4:
                        extra_yaw, extra_pitch = 0, -90
                    else:
                        extra_yaw, extra_pitch = 0, 90
                    pixel_data, mask = self.lidar_to_camera_transform(
                        camera_reference_points_real,
                        offset_x, offset_y, offset_z,
                        extra_pitch, extra_yaw, roll_z,
                        rotation_lidar_2_camera,
                        real_camera_intrinsic)

                camera_reference_points_list.append(paddle.unsqueeze(pixel_data, axis=2))
                camera_mask_list.append(paddle.unsqueeze(mask, axis=2))
                camera_mask_list[i] = camera_mask_list[i].unsqueeze(2).unsqueeze(3).expand(
                    [bs, Len_q, n_head, self.num_levels // 2, n_num, 2])

                camera_sampling_locations = camera_reference_points_list[i].expand(
                    [-1, -1, self.num_levels // 2, -1]).reshape([
                        bs, Len_q, 1, self.num_levels // 2, 1, 2
                    ]) + sampling_offsets[:, :, :, num_radar_levels:, :, :] / offset_normalizer
                camera_sampling_locations_list.append(camera_sampling_locations)

        if self.visualize:
            self._maybe_visualize(
                gt_meta,
                reference_points,
                radar_sampling_locations,
                camera_reference_points_list,
                camera_sampling_locations_list,
                camera_mask_list,
                radar_value_spatial_shapes,
                camera_value_spatial_shapes,
                angle,
                attention_weights)

        radar_value_spatial_shapes = paddle.to_tensor(radar_value_spatial_shapes)
        radar_value_level_start_index = paddle.to_tensor(radar_value_level_start_index)
        camera_value_spatial_shapes = paddle.to_tensor(camera_value_spatial_shapes)
        camera_value_level_start_index = paddle.to_tensor(camera_value_level_start_index)

        output = self.ms_deformable_attn_core(
            radar_value, camera_value, radar_value_spatial_shapes, radar_value_level_start_index,
            camera_value_spatial_shapes, camera_value_level_start_index,
            radar_sampling_locations, camera_sampling_locations_list, camera_mask_list, attention_weights)
        output = self.output_proj(output)

        return output
class PPMSDeformableAttention_Missing(MSDeformableAttention_Missing):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                eval_all = False):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        num_quires = 300

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        if self.training or eval_all:
            Len_q_dnvisir = Len_q - num_quires * 2
            query_dnvisir = query[:,:-num_quires * 2,:]
            sampling_offsets = self.sampling_offsets(query_dnvisir).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query_dnvisir).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points])

            query_vis = query[:,-num_quires * 2:-num_quires,:]
            sampling_offsets_vis = self.sampling_offsets_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points, 2])
            attention_weights_vis = self.attention_weights_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2 * self.num_points])
            attention_weights_vis = F.softmax(attention_weights_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points])

            query_ir = query[:,-num_quires:,:]
            sampling_offsets_ir = self.sampling_offsets_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points, 2])
            attention_weights_ir = self.attention_weights_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2 * self.num_points])
            attention_weights_ir = F.softmax(attention_weights_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points])
        else:
            sampling_offsets = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            if self.training or eval_all:
                sampling_locations = (
                reference_points[:, :-num_quires * 2, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :-num_quires * 2, None, :, None, 2:] *
                0.5)

                sampling_locations_vis = (
                        reference_points[:, -num_quires * 2:-num_quires, None, :, None, :2] + sampling_offsets_vis /
                        self.num_points * reference_points[:, -num_quires * 2:-num_quires, None, :, None, 2:] *
                        0.5)

                sampling_locations_ir = (
                        reference_points[:, -num_quires:, None, :, None, :2] + sampling_offsets_ir /
                        self.num_points * reference_points[:, -num_quires:, None, :, None, 2:] *
                        0.5)
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            if self.training or eval_all:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_spatial_shapes_vis = value_spatial_shapes[:3,:]
                value_spatial_shapes_ir = value_spatial_shapes[:3,:]
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                value_level_start_index_vis = value_level_start_index[0:3,]
                value_level_start_index_ir = value_level_start_index[0:3,]


                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
                output_vis = self.ms_deformable_attn_core(
                    value[:,:Len_v//2,:,:], value_spatial_shapes_vis, value_level_start_index_vis,
                    sampling_locations_vis, attention_weights_vis)
                output_ir = self.ms_deformable_attn_core(
                    value[:, Len_v//2:, :, :], value_spatial_shapes_ir, value_level_start_index_ir,
                    sampling_locations_ir, attention_weights_ir)
                output = paddle.concat([output,output_vis,output_ir], axis=1)
            else:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class PPMSDeformableAttention_Missing_V2(MSDeformableAttention):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                # vis_value,
                # ir_value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                eval_all = False):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        num_quires = 300

        value = self.value_proj(value)
        # vis_value = self.value_proj(vis_value)
        # ir_value = self.value_proj(ir_value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        # vis_value = vis_value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        # ir_value = ir_value.reshape([bs, Len_v, self.num_heads, self.head_dim])


        if self.training or eval_all:
            Len_q_dnvisir = Len_q - num_quires * 2
            # query_dnvisir = query[:,:-num_quires * 2,:]
            sampling_offsets = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

            sampling_offsets_dnvisir = sampling_offsets[:,:-num_quires*2,:,:,:,:]
            attention_weights_dnvisir = attention_weights[:,:-num_quires*2,:,:,:]

            sampling_offsets_vis = sampling_offsets[:,-num_quires*2:-num_quires,:,:,:,:]
            attention_weights_vis = attention_weights[:, -num_quires * 2:-num_quires, :, :, :]

            sampling_offsets_ir = sampling_offsets[:, -num_quires:, :, :, :, :]
            attention_weights_ir = attention_weights[:, -num_quires:, :, :, :]
            # query_vis = query[:,-num_quires * 2:-num_quires,:]
            # sampling_offsets_vis = self.sampling_offsets_vis(query_vis).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points, 2])
            # attention_weights_vis = self.attention_weights_vis(query_vis).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels//2 * self.num_points])
            # attention_weights_vis = F.softmax(attention_weights_vis).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points])
            #
            # query_ir = query[:,-num_quires:,:]
            # sampling_offsets_ir = self.sampling_offsets_ir(query_ir).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points, 2])
            # attention_weights_ir = self.attention_weights_ir(query_ir).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels // 2 * self.num_points])
            # attention_weights_ir = F.softmax(attention_weights_ir).reshape(
            #     [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points])
        else:
            sampling_offsets_dnvisir = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights_dnvisir = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights_dnvisir = F.softmax(attention_weights_dnvisir).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            if self.training or eval_all:
                sampling_locations = (
                reference_points[:, :-num_quires * 2, None, :, None, :2] + sampling_offsets_dnvisir /
                self.num_points * reference_points[:, :-num_quires * 2, None, :, None, 2:] *
                0.5)

                sampling_locations_vis = (
                        reference_points[:, -num_quires * 2:-num_quires, None, :, None, :2] + sampling_offsets_vis /
                        self.num_points * reference_points[:, -num_quires * 2:-num_quires, None, :, None, 2:] *
                        0.5)

                sampling_locations_ir = (
                        reference_points[:, -num_quires:, None, :, None, :2] + sampling_offsets_ir /
                        self.num_points * reference_points[:, -num_quires:, None, :, None, 2:] *
                        0.5)
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            if self.training or eval_all:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                # value_spatial_shapes_vis = value_spatial_shapes[:3,:]
                # value_spatial_shapes_ir = value_spatial_shapes[:3,:]
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                # value_level_start_index_vis = value_level_start_index[0:3,]
                # value_level_start_index_ir = value_level_start_index[0:3,]


                output = self.ms_deformable_attn_core(
                    value[:,:Len_v//2,:,:], value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights_dnvisir)
                output_vis = self.ms_deformable_attn_core(
                    paddle.concat([value[:,:Len_v//4,:,:],value[:,-Len_v//4:,:,:]],axis=1),
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_vis, attention_weights_vis)
                output_ir = self.ms_deformable_attn_core(
                    paddle.concat([value[:,Len_v//2:3*Len_v//4,:,:],value[:,Len_v//4:Len_v//2,:,:]],axis=1),
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_ir, attention_weights_ir)
                output = paddle.concat([output,output_vis,output_ir], axis=1)
            else:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

class PPMSDeformableAttention_Missing_V3(MSDeformableAttention):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                vis_value,
                ir_value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                eval_all = False):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        num_quires = 300

        value = self.value_proj(value)
        vis_value = self.value_proj(vis_value)
        ir_value = self.value_proj(ir_value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        vis_value = vis_value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        ir_value = ir_value.reshape([bs, Len_v, self.num_heads, self.head_dim])


        if self.training or eval_all:
            Len_q_dnvisir = Len_q - num_quires * 2
            # query_dnvisir = query[:,:-num_quires * 2,:]
            sampling_offsets = self.sampling_offsets(query[:,:-2*num_quires,:]).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query[:,:-2*num_quires,:]).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points])

            # sampling_offsets_dnvisir = sampling_offsets[:,:-num_quires*2,:,:,:,:]
            # attention_weights_dnvisir = attention_weights[:,:-num_quires*2,:,:,:]
            #
            # sampling_offsets_vis = sampling_offsets[:,-num_quires*2:-num_quires,:,:,:,:]
            # attention_weights_vis = attention_weights[:, -num_quires * 2:-num_quires, :, :, :]
            #
            # sampling_offsets_ir = sampling_offsets[:, -num_quires:, :, :, :, :]
            # attention_weights_ir = attention_weights[:, -num_quires:, :, :, :]
            query_vis = query[:,-num_quires * 2:-num_quires,:]
            sampling_offsets_vis = self.sampling_offsets(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights_vis = self.attention_weights(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels * self.num_points])
            attention_weights_vis = F.softmax(attention_weights_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels, self.num_points])

            query_ir = query[:,-num_quires:,:]
            sampling_offsets_ir = self.sampling_offsets(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels , self.num_points, 2])
            attention_weights_ir = self.attention_weights(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels  * self.num_points])
            attention_weights_ir = F.softmax(attention_weights_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels , self.num_points])
        else:
            sampling_offsets = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            if self.training or eval_all:
                sampling_locations = (
                reference_points[:, :-num_quires * 2, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :-num_quires * 2, None, :, None, 2:] *
                0.5)

                sampling_locations_vis = (
                        reference_points[:, -num_quires * 2:-num_quires, None, :, None, :2] + sampling_offsets_vis /
                        self.num_points * reference_points[:, -num_quires * 2:-num_quires, None, :, None, 2:] *
                        0.5)

                sampling_locations_ir = (
                        reference_points[:, -num_quires:, None, :, None, :2] + sampling_offsets_ir /
                        self.num_points * reference_points[:, -num_quires:, None, :, None, 2:] *
                        0.5)
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # ## sample location of different semantic level visulize
        #
        # ##plot location point
        # bs = len(gt_meta['im_id'])
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h, w, _ = vis_imgs[0].shape
        # real_hw = [w, h]
        # real_hw = np.array(real_hw)
        #
        # #for visir
        # # real_location_point = np.array(sampling_locations) * real_hw
        # # real_weights = np.array(attention_weights)
        #
        # #for vis
        #
        # #for ir
        # real_location_point = np.array(sampling_locations_ir) * real_hw
        # real_weights = np.array(attention_weights_ir)
        #
        # radius = 4
        # color_r = (0, 0, 255)
        # color_b = (230, 216, 173)
        # color_g = (152, 251, 152)
        # min_score = 0.0
        # max_score = 0.15
        # yellow_color = (0, 255, 255)
        # red_color = (0, 0, 255)
        #
        # low_blue = (230, 216, 173)
        # low_green = (152, 251, 152)
        # low_purple = (186,85,211)
        # low_red = (128,128,240)
        # blue = (255,0,0)
        # green = (0,255,0)
        # purple = (128,0,128)
        # red = (0,0,255)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_score[xx][ii] > 0:
        #             # if topk_ind_mask[xx][ii] == 1:
        #             for mm in range(8):
        #                 for zz in range(3):
        #                     for pp in range(4):
        #                         if real_weights[xx][ii][mm][zz][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                                       round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(vis_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value = int(255*(real_weights[xx][ii][mm][zz][pp]-min_score)/(max_score - min_score))
        #                         # color = tuple(int((1 - real_weights[xx][ii][mm][zz][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius-2,color,-1)
        #
        #                         # else:
        #                         #     cv2.circle(vis_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #
        #                         if real_weights[xx][ii][mm][zz + 3][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz + 3][pp][0]),
        #                                                      round(real_location_point[xx][ii][mm][zz + 3][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz + 3][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(ir_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz + 3][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz + 3][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value2 = int(
        #                         #     255 * (real_weights[xx][ii][mm][zz + 2][pp] - min_score) / (max_score - min_score))
        #                         # color2 = tuple(int((1 - real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #            radius - 2, color2, -1)
        #
        #                         # else:
        #                         #     cv2.circle(ir_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #             #####################no no no
        #             # elif topk_ind_mask[xx][ii] == 2:
        #             # for mm in range(8):
        #             # for zz in range(2):
        #             #     for pp in range(4):
        #             #         if real_weights[xx][ii][mm][zz+2][pp] > 0.1:
        #             #             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                                       round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 1, color_r, -1)
        #             #         elif real_weights[xx][ii][mm][zz+2][pp] > 0.05:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_g, -1)
        #             #         else:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_b, -1)
        #
        # for ii in range(bs):
        #     # heatmap_image1 = cv2.applyColorMap(vis_imgs[ii],cv2.COLORMAP_JET)
        #     # heatmap_image2 = cv2.applyColorMap(ir_imgs[ii],cv2.COLORMAP_JET)
        #     cv2.imwrite(
        #         '/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/sampling_points_ir/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0] + '_vis.png', vis_imgs[ii])
        #     cv2.imwrite(
        #         '/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/sampling_points_ir/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0] + '_ir.png', ir_imgs[ii])


        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            if self.training or eval_all:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                # value_spatial_shapes_vis = value_spatial_shapes[:3,:]
                # value_spatial_shapes_ir = value_spatial_shapes[:3,:]
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                # value_level_start_index_vis = value_level_start_index[0:3,]
                # value_level_start_index_ir = value_level_start_index[0:3,]


                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
                output_vis = self.ms_deformable_attn_core(
                    vis_value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_vis, attention_weights_vis)
                output_ir = self.ms_deformable_attn_core(
                    ir_value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_ir, attention_weights_ir)
                output = paddle.concat([output,output_vis,output_ir], axis=1)
            else:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

class PPMSDeformableAttention_Missing_V3_3samplinglayer(MSDeformableAttention_Missing_3samplinglayer):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                vis_value,
                ir_value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                eval_all = False):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        num_quires = 300

        value = self.value_proj(value)
        vis_value = self.value_proj(vis_value)
        ir_value = self.value_proj(ir_value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        vis_value = vis_value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        ir_value = ir_value.reshape([bs, Len_v, self.num_heads, self.head_dim])


        if self.training or eval_all:
            Len_q_dnvisir = Len_q - num_quires * 2
            # query_dnvisir = query[:,:-num_quires * 2,:]
            sampling_offsets = self.sampling_offsets(query[:,:-2*num_quires,:]).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query[:,:-2*num_quires,:]).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points])

            # sampling_offsets_dnvisir = sampling_offsets[:,:-num_quires*2,:,:,:,:]
            # attention_weights_dnvisir = attention_weights[:,:-num_quires*2,:,:,:]
            #
            # sampling_offsets_vis = sampling_offsets[:,-num_quires*2:-num_quires,:,:,:,:]
            # attention_weights_vis = attention_weights[:, -num_quires * 2:-num_quires, :, :, :]
            #
            # sampling_offsets_ir = sampling_offsets[:, -num_quires:, :, :, :, :]
            # attention_weights_ir = attention_weights[:, -num_quires:, :, :, :]
            query_vis = query[:,-num_quires * 2:-num_quires,:]
            sampling_offsets_vis = self.sampling_offsets_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights_vis = self.attention_weights_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels * self.num_points])
            attention_weights_vis = F.softmax(attention_weights_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels, self.num_points])

            query_ir = query[:,-num_quires:,:]
            sampling_offsets_ir = self.sampling_offsets_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels , self.num_points, 2])
            attention_weights_ir = self.attention_weights_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels  * self.num_points])
            attention_weights_ir = F.softmax(attention_weights_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels , self.num_points])
        else:
            sampling_offsets = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            if self.training or eval_all:
                sampling_locations = (
                reference_points[:, :-num_quires * 2, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :-num_quires * 2, None, :, None, 2:] *
                0.5)

                sampling_locations_vis = (
                        reference_points[:, -num_quires * 2:-num_quires, None, :, None, :2] + sampling_offsets_vis /
                        self.num_points * reference_points[:, -num_quires * 2:-num_quires, None, :, None, 2:] *
                        0.5)

                sampling_locations_ir = (
                        reference_points[:, -num_quires:, None, :, None, :2] + sampling_offsets_ir /
                        self.num_points * reference_points[:, -num_quires:, None, :, None, 2:] *
                        0.5)
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # sample location of different semantic level visulize

        # ##plot location point
        # bs = len(gt_meta['im_id'])
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h, w, _ = vis_imgs[0].shape
        # real_hw = [w, h]
        # real_hw = np.array(real_hw)
        #
        # #for visir
        # # real_location_point = np.array(sampling_locations) * real_hw
        # # real_weights = np.array(attention_weights)
        #
        # #for vis
        #
        # #for ir
        # real_location_point = np.array(sampling_locations_ir) * real_hw
        # real_weights = np.array(attention_weights_ir)
        #
        # radius = 4
        # color_r = (0, 0, 255)
        # color_b = (230, 216, 173)
        # color_g = (152, 251, 152)
        # min_score = 0.0
        # max_score = 0.15
        # yellow_color = (0, 255, 255)
        # red_color = (0, 0, 255)
        #
        # low_blue = (230, 216, 173)
        # low_green = (152, 251, 152)
        # low_purple = (186,85,211)
        # low_red = (128,128,240)
        # blue = (255,0,0)
        # green = (0,255,0)
        # purple = (128,0,128)
        # red = (0,0,255)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_score[xx][ii] > 0:
        #             # if topk_ind_mask[xx][ii] == 1:
        #             for mm in range(8):
        #                 for zz in range(3):
        #                     for pp in range(4):
        #                         if real_weights[xx][ii][mm][zz][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                                       round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(vis_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value = int(255*(real_weights[xx][ii][mm][zz][pp]-min_score)/(max_score - min_score))
        #                         # color = tuple(int((1 - real_weights[xx][ii][mm][zz][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(vis_imgs[xx], (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius-2,color,-1)
        #
        #                         # else:
        #                         #     cv2.circle(vis_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #
        #                         if real_weights[xx][ii][mm][zz + 3][pp] > 0.1:
        #                             if zz == 0:
        #                                 CColor = blue
        #                             elif zz == 1:
        #                                 CColor = green
        #                             elif zz == 2:
        #                                 CColor = red
        #                             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz + 3][pp][0]),
        #                                                      round(real_location_point[xx][ii][mm][zz + 3][pp][1])),
        #                                        radius - 2, CColor, -1)
        #                         elif real_weights[xx][ii][mm][zz + 3][pp] > 0.04:
        #                             if zz == 0:
        #                                 CColor = low_blue
        #                             elif zz == 1:
        #                                 CColor = low_green
        #                             elif zz == 2:
        #                                 CColor = low_red
        #                             cv2.circle(ir_imgs[xx],
        #                                        (round(real_location_point[xx][ii][mm][zz + 3][pp][0]),
        #                                         round(real_location_point[xx][ii][mm][zz + 3][pp][1])),
        #                                        radius - 3, CColor, -1)
        #
        #                         # color_value2 = int(
        #                         #     255 * (real_weights[xx][ii][mm][zz + 2][pp] - min_score) / (max_score - min_score))
        #                         # color2 = tuple(int((1 - real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * channel + (real_weights[xx][ii][mm][zz + 2][pp] / 0.15) * red_color[i]) for i, channel in enumerate(yellow_color))
        #                         # cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                           round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #            radius - 2, color2, -1)
        #
        #                         # else:
        #                         #     cv2.circle(ir_imgs[xx],
        #                         #                (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #                         #                 round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #                         #                radius - 3, color_b, -1)
        #             #####################no no no
        #             # elif topk_ind_mask[xx][ii] == 2:
        #             # for mm in range(8):
        #             # for zz in range(2):
        #             #     for pp in range(4):
        #             #         if real_weights[xx][ii][mm][zz+2][pp] > 0.1:
        #             #             cv2.circle(ir_imgs[xx], (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                                       round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 1, color_r, -1)
        #             #         elif real_weights[xx][ii][mm][zz+2][pp] > 0.05:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_g, -1)
        #             #         else:
        #             #             cv2.circle(ir_imgs[xx],
        #             #                        (round(real_location_point[xx][ii][mm][zz+2][pp][0]),
        #             #                         round(real_location_point[xx][ii][mm][zz+2][pp][1])),
        #             #                        radius - 3, color_b, -1)
        #
        # for ii in range(bs):
        #     # heatmap_image1 = cv2.applyColorMap(vis_imgs[ii],cv2.COLORMAP_JET)
        #     # heatmap_image2 = cv2.applyColorMap(ir_imgs[ii],cv2.COLORMAP_JET)
        #     cv2.imwrite(
        #         '/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/sampling_points_ir/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0] + '_vis.png', vis_imgs[ii])
        #     cv2.imwrite(
        #         '/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/sampling_points_ir/' + 'layer' + str(
        #             gt_meta['layer'] + 1) + '/' +
        #         gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0] + '_ir.png', ir_imgs[ii])


        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            if self.training or eval_all:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                # value_spatial_shapes_vis = value_spatial_shapes[:3,:]
                # value_spatial_shapes_ir = value_spatial_shapes[:3,:]
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                # value_level_start_index_vis = value_level_start_index[0:3,]
                # value_level_start_index_ir = value_level_start_index[0:3,]


                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
                output_vis = self.ms_deformable_attn_core(
                    vis_value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_vis, attention_weights_vis)
                output_ir = self.ms_deformable_attn_core(
                    ir_value,
                    value_spatial_shapes, value_level_start_index,
                    sampling_locations_ir, attention_weights_ir)
                output = paddle.concat([output,output_vis,output_ir], axis=1)
            else:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

class PPMSDeformableAttention_Missing_Groupx5(MSDeformableAttention_Missing):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None,
                eval_all = False):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        num_quires = 300
        value = self.value_proj(value)
        value_fuse = paddle.split(value,2,axis=1)[0] + paddle.split(value,2,axis=1)[1]
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        value_fuse = value_fuse.reshape([bs, Len_v//2, self.num_heads, self.head_dim])

        if self.training or eval_all:
            Len_q_dnvisir = Len_q - num_quires * 4
            query_dnvisir = query[:,:-num_quires * 4,:]
            sampling_offsets = self.sampling_offsets(query_dnvisir).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query_dnvisir).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q_dnvisir, self.num_heads, self.num_levels, self.num_points])

            query_vis = query[:,-num_quires * 2:-num_quires,:]
            sampling_offsets_vis = self.sampling_offsets_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points, 2])
            attention_weights_vis_temp = self.attention_weights_vis(query_vis).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2 * self.num_points])
            attention_weights_vis = F.softmax(attention_weights_vis_temp).reshape(
                [bs, num_quires, self.num_heads, self.num_levels//2, self.num_points])

            query_ir = query[:,-num_quires:,:]
            sampling_offsets_ir = self.sampling_offsets_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points, 2])
            attention_weights_ir_temp = self.attention_weights_ir(query_ir).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2 * self.num_points])
            attention_weights_ir = F.softmax(attention_weights_ir_temp).reshape(
                [bs, num_quires, self.num_heads, self.num_levels // 2, self.num_points])
        else:
            sampling_offsets = self.sampling_offsets(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
            attention_weights = self.attention_weights(query).reshape(
                [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
            attention_weights = F.softmax(attention_weights).reshape(
                [bs, Len_q, self.num_heads, self.num_levels, self.num_points])


        ##test
        # if sort_index is not None:
        #     attention_weights_visul = paddle.gather_nd(attention_weights, sort_index)[:, :30]
        #     attention_weights_visul = attention_weights_visul[:,:,:,:,0] +attention_weights_visul[:,:,:,:,1] +attention_weights_visul[:,:,:,:,2] + attention_weights_visul[:,:,:,:,3]
        #     attention_weights_visul_np = np.array(attention_weights_visul)
        #     attention_weights_visul_np_vis = attention_weights_visul_np[:,:,:,0]+attention_weights_visul_np[:,:,:,1]+attention_weights_visul_np[:,:,:,2]
        #     attention_weights_visul_np_ir = attention_weights_visul_np[:, :, :, 3] + attention_weights_visul_np[:, :,
        #                                                                               :,
        #                                                                               4] + attention_weights_visul_np[:,
        #                                                                                    :, :, 5]
        ##

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            if self.training or eval_all:
                sampling_locations = (
                reference_points[:, :-num_quires * 4, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :-num_quires * 4, None, :, None, 2:] *
                0.5)

                sampling_locations_vis = (
                        reference_points[:, -num_quires * 2:-num_quires, None, :, None, :2] + sampling_offsets_vis /
                        self.num_points * reference_points[:, -num_quires * 2:-num_quires, None, :, None, 2:] *
                        0.5)

                sampling_locations_ir = (
                        reference_points[:, -num_quires:, None, :, None, :2] + sampling_offsets_ir /
                        self.num_points * reference_points[:, -num_quires:, None, :, None, 2:] *
                        0.5)
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] *
                    0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            if self.training or eval_all:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_spatial_shapes_vis = value_spatial_shapes[:3,:]
                value_spatial_shapes_ir = value_spatial_shapes[:3,:]
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                value_level_start_index_vis = value_level_start_index[0:3,]
                value_level_start_index_ir = value_level_start_index[0:3,]

                # sampling_locations_fuse_visir = paddle.concat([sampling_locations_vis,sampling_locations_ir],axis=3)
                # attention_weights_fuse_visir = paddle.concat([attention_weights_vis_temp,attention_weights_ir_temp],axis=3)
                # attention_weights_fuse_visir = F.softmax(attention_weights_fuse_visir).reshape(
                #     [bs, num_quires, self.num_heads, self.num_levels , self.num_points])

                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
                output_vis = self.ms_deformable_attn_core(
                    value[:,:Len_v//2,:,:], value_spatial_shapes_vis, value_level_start_index_vis,
                    sampling_locations_vis, attention_weights_vis)
                output_ir = self.ms_deformable_attn_core(
                    value[:, Len_v//2:, :, :], value_spatial_shapes_ir, value_level_start_index_ir,
                    sampling_locations_ir, attention_weights_ir)
                output_fuse_vis = self.ms_deformable_attn_core(
                    value_fuse, value_spatial_shapes_vis, value_level_start_index_vis,
                    sampling_locations_vis, attention_weights_vis)
                output_fuse_ir = self.ms_deformable_attn_core(
                    value_fuse, value_spatial_shapes_ir, value_level_start_index_ir,
                    sampling_locations_ir, attention_weights_ir)

                output = paddle.concat([output,output_fuse_vis,output_fuse_ir,output_vis,output_ir], axis=1)
            else:
                value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
                value_level_start_index = paddle.to_tensor(value_level_start_index)
                output = self.ms_deformable_attn_core(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

class PPMSDeformableAttention_key_aware(MSDeformableAttention_key_aware):
    def forward(self,
                gt_meta,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        key = value
        key = self.key_proj(key)
        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        # attention_weights = self.attention_weights(query).reshape(
        #     [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        # attention_weights = F.softmax(attention_weights).reshape(
        #     [bs, Len_q, self.num_heads, self.num_levels, self.num_points])
        attention_weights = None

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.to_tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))
        if not isinstance(query, paddle.Tensor):
            from ppdet.modeling.transformers.utils import deformable_attention_core_func
            output = deformable_attention_core_func(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        else:
            query = self.query_proj(query)

            value_spatial_shapes = paddle.to_tensor(value_spatial_shapes)
            value_level_start_index = paddle.to_tensor(value_level_start_index)
            output = self.ms_key_aware_deformable_attn_core(
                query,value,key, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None,
                 key_aware=False):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        if key_aware:
            self.cross_attn = PPMSDeformableAttention_key_aware(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        else:
            self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderLayer_RadarCamera_Rotate(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 angle_max=None,
                 angle_proj=None,
                 weight_attr=None,
                 bias_attr=None,
                 key_aware=False):
        super(TransformerDecoderLayer_RadarCamera_Rotate, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.cross_attn = PPMSDeformableAttention_RadarCamera_Rotate(
            d_model, n_head, n_levels, n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                angle,
                radar_memory,
                camera_memory,
                radar_memory_spatial_shapes,
                radar_memory_level_start_index,
                camera_memory_spatial_shapes,
                camera_memory_level_start_index,
                angle_max,
                half_pi_bin,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask=None,
                topk_score=None,
                mask_vis=None):
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            gt_meta,
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            angle,
            radar_memory,
            camera_memory,
            radar_memory_spatial_shapes,
            radar_memory_level_start_index,
            camera_memory_spatial_shapes,
            camera_memory_level_start_index,
            angle_max,
            half_pi_bin,
            memory_mask,
            topk_ind_mask,
            topk_score,
            mask_vis=mask_vis)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderLayer_RadarCamera(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None,
                 key_aware=False):
        super(TransformerDecoderLayer_RadarCamera, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention

        self.cross_attn = PPMSDeformableAttention_RadarCamera(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                radar_memory,
                camera_memory,
                radar_memory_spatial_shapes,
                radar_memory_level_start_index,
                camera_memory_spatial_shapes,
                camera_memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, radar_memory, camera_memory,
            radar_memory_spatial_shapes, radar_memory_level_start_index,
                               camera_memory_spatial_shapes, camera_memory_level_start_index,memory_mask,topk_ind_mask,topk_score)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_Group(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_Group, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*2,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*2:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*2,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 2:, :]
            k_dnvisir = k[:,:-num_query*2,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 2:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,2,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,2,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 2, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 2, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_Groupx3_Missing(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_Groupx3_Missing, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention_Missing(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*2,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*2:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*2,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 2:, :]
            k_dnvisir = k[:,:-num_query*2,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 2:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,2,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,2,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 2, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 2, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score,eval_all = eval_all)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_Groupx3_Missing_V2(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_Groupx3_Missing_V2, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention_Missing_V2(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                # vis_memory,
                # ir_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*2,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*2:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*2,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 2:, :]
            k_dnvisir = k[:,:-num_query*2,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 2:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,2,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,2,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 2, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 2, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score,eval_all = eval_all)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderLayer_Groupx3_Missing_V3(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_Groupx3_Missing_V3, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention_Missing_V3_3samplinglayer(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                vis_memory,
                ir_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*2,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*2:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*2,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 2:, :]
            k_dnvisir = k[:,:-num_query*2,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 2:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,2,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,2,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 2, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 2, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,vis_memory,ir_memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score,eval_all = eval_all)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_Groupx5_Missing(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_Groupx5_Missing, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention_Missing_Groupx5(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*4,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*4:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*4,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 4:, :]
            k_dnvisir = k[:,:-num_query*4,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 4:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,4,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,4,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 4, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 4, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score,eval_all = eval_all)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_Groupx4(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None,
                 key_aware=False):
        super(TransformerDecoderLayer_Groupx4, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        if key_aware:
            self.cross_attn = PPMSDeformableAttention_key_aware(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        else:
            self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                      n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                eval_all = False):
        # memory_spatial_shapes = paddle.to_tensor(memory_spatial_shapes)
        # memory_level_start_index = paddle.to_tensor(memory_level_start_index)
        # self attention
        num_query = topk_score.shape[1]
        bs = topk_score.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

        if self.training or eval_all is True:
            tgt_dnvisir = tgt[:,:-num_query*3,:]  # [bs,dn+n_query,256]
            # tgt_vis = tgt[:,-num_query*2:-num_query,:]
            # tgt_ir = tgt[:,-num_query:,:]
            tgt_vis_ir = tgt[:,-num_query*3:,:]  #[bs,n_query*2,256]
            q_dnvisir = q[:,:-num_query*3,:]
            # q_vis = q[:,-num_query*2:-num_query,:]
            # q_ir = q[:,-num_query:,:]
            q_vis_ir = q[:, -num_query * 3:, :]
            k_dnvisir = k[:,:-num_query*3,:]
            # k_vis = k[:,-num_query*2:-num_query,:]
            # k_ir = k[:,-num_query:,:]
            k_vis_ir = k[:, -num_query * 3:, :]

            tgt_vis_ir = paddle.concat(paddle.split(tgt_vis_ir,3,axis=1),axis=0)
            q_vis_ir = paddle.concat(paddle.split(q_vis_ir,3,axis=1),axis=0)
            k_vis_ir = paddle.concat(paddle.split(k_vis_ir, 3, axis=1), axis=0)
            #do dnvisir self attention
            tgt2_dnvisir = self.self_attn(q_dnvisir, k_dnvisir,value=tgt_dnvisir, attn_mask=attn_mask)
            #do vis and ir self attention
            #[2*bs,n_query,256]
            tgt2_vis_ir = self.self_attn(q_vis_ir,k_vis_ir,value=tgt_vis_ir,attn_mask=None)
            #to origin shape
            tgt2_vis_ir = paddle.concat(paddle.split(tgt2_vis_ir, 3, axis=0), axis=1)

            tgt2 = paddle.concat([tgt2_dnvisir,tgt2_vis_ir],axis=1)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)



        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_BA(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_BA, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = PPMSDeformableAttention(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                sort_index = None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask,topk_ind_mask,topk_score,sort_index)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderLayer_split(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer_split, self).__init__()

        # self attention
        self.self_attn_vis = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.self_attn_ir = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn_vis = PPMSDeformableAttention(d_model, n_head, n_levels//2,
                                                  n_points, 1.0)
        self.cross_attn_ir = PPMSDeformableAttention(d_model, n_head, n_levels//2,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                num_queries,
                gt_meta,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))

            #vis query self attention
        q_vis = paddle.concat([q[:,:(q.shape[1] - num_queries)//2,:],q[:,-num_queries:-num_queries//2,:]],axis=1)
        k_vis = paddle.concat([k[:,:(q.shape[1] - num_queries)//2,:],k[:,-num_queries:-num_queries//2,:]],axis=1)
        v_vis = paddle.concat([tgt[:,:(q.shape[1] - num_queries)//2,:],tgt[:,-num_queries:-num_queries//2,:]],axis=1)
        q_ir = paddle.concat([q[:,(q.shape[1] - num_queries)//2:(q.shape[1] - num_queries),:],q[:,-num_queries//2:,:]],axis=1)
        k_ir = paddle.concat([k[:,(q.shape[1] - num_queries)//2:(q.shape[1] - num_queries),:],k[:,-num_queries//2:,:]],axis=1)
        v_ir = paddle.concat([tgt[:,(q.shape[1] - num_queries)//2:(q.shape[1] - num_queries),:],tgt[:,-num_queries//2:,:]],axis=1)
        tgt2_1 = self.self_attn_vis(q_vis, k_vis, value=v_vis, attn_mask=attn_mask)
        tgt2_2 = self.self_attn_ir(q_ir, k_ir, value=v_ir, attn_mask=attn_mask)
        #tgt2 = paddle.concat([tgt2_1,tgt2_2],axis=1) #(vis-dn,vis,ir-dn,ir)
        tgt2 = paddle.concat([tgt2_1[:,:tgt2_1.shape[1]-num_queries//2,:],tgt2_2[:,:tgt2_2.shape[1]-num_queries//2,:],
                              tgt2_1[:,-num_queries//2:,:],tgt2_2[:,-num_queries//2:,:]],axis=1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        ##
        attentioned_tgt = self.with_pos_embed(tgt, query_pos_embed)
        attentioned_tgt_vis = paddle.concat([attentioned_tgt[:,:(q.shape[1] - num_queries)//2,:],attentioned_tgt[:,-num_queries:-num_queries//2,:]],axis=1)
        reference_points_vis = paddle.concat([reference_points[:,:(q.shape[1] - num_queries)//2,:],reference_points[:,-num_queries:-num_queries//2,:]],axis=1)
        attentioned_tgt_ir = paddle.concat([attentioned_tgt[:,(q.shape[1] - num_queries)//2:(q.shape[1] - num_queries),:],attentioned_tgt[:,-num_queries//2:,:]],axis=1)
        reference_points_ir = paddle.concat([reference_points[:,(q.shape[1] - num_queries)//2:(q.shape[1] - num_queries),:],reference_points[:,-num_queries//2:,:]],axis=1)

        # cross attention
        tgt2_1 = self.cross_attn_vis(gt_meta,
            attentioned_tgt_vis, reference_points_vis, memory[:,:memory_level_start_index[3],:],
            memory_spatial_shapes[:3], memory_level_start_index[:3], memory_mask,topk_ind_mask,topk_score)
        tgt2_2 = self.cross_attn_ir(gt_meta,
            attentioned_tgt_ir, reference_points_ir, memory[:,memory_level_start_index[3]:,:],
            memory_spatial_shapes[:3], memory_level_start_index[:3], memory_mask,topk_ind_mask,topk_score)
        tgt2 = paddle.concat(
            [tgt2_1[:, :tgt2_1.shape[1] - num_queries // 2, :], tgt2_2[:, :tgt2_2.shape[1] - num_queries // 2, :],
             tgt2_1[:, -num_queries // 2:, :], tgt2_2[:, -num_queries // 2:, :]], axis=1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)


        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])






            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_RadarCamera(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_RadarCamera, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                gt_meta,
                tgt_radar,
                ref_points_unact,
                radar_memory,
                camera_memory,
                radar_memory_spatial_shapes,
                radar_memory_level_start_index,
                camera_memory_spatial_shapes,
                camera_memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None):
        output = tgt_radar
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)


        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, radar_memory, camera_memory,
                           radar_memory_spatial_shapes, radar_memory_level_start_index,
                           camera_memory_spatial_shapes, camera_memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])






            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)

class TransformerDecoder_Rotate(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, angle_max, angle_proj, half_pi_bin,eval_idx=-1):
        super(TransformerDecoder_Rotate, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.angle_max = angle_max
        self.angle_proj = angle_proj
        self.half_pi_bin = half_pi_bin

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                ref_angle_cls,
                ref_angle,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                angle_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                mask_vis = None,
                return_last_query=False):

        # if self.training:
        #     bs, len_qd, d = tgt.shape
        #     _,len_q = topk_ind_mask.shape
        #     len_denosing = len_qd - len_q
        #     denosing_mask = paddle.concat([paddle.ones([len_denosing // 2], dtype='bool'), paddle.zeros([len_denosing // 2], dtype='bool')])
        #     denosing_mask = paddle.unsqueeze(denosing_mask, axis=0)
        #     denosing_mask = paddle.unsqueeze(denosing_mask, axis=-1)
        #     denosing_mask = paddle.broadcast_to(denosing_mask, shape=[bs, len_denosing, 4])
        #
        #     mask_vis = paddle.concat([denosing_mask, mask_vis], axis=1)
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_angles_cls = []
        dec_out_angles = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        # #get angle
        b, l = ref_angle.shape[:2]
        # angle = F.softmax(ref_angle_cls.reshape([b, l, 1, self.angle_max + 1
        #                                       ])).matmul(self.angle_proj)
        angle = ref_angle

        gt_meta['layer'] = 0

        #prof = Profiler(targets=[paddle.profiler.ProfilerTarget.GPU])
        # prof = Profiler(targets=[paddle.profiler.ProfilerTarget.GPU])
        # with RecordEvent('code_to_analyze'):

        for i, layer in enumerate(self.layers):


            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, angle, memory,
                           memory_spatial_shapes, memory_level_start_index,self.angle_max, self.half_pi_bin,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score, mask_vis)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))
            angle_cls = angle_head[i](output)
            angle = F.softmax(angle_cls.reshape([b, l, 1, self.angle_max + 1
                                              ])).matmul(self.angle_proj)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_angles_cls.append(angle_cls)
                dec_out_angles.append(angle)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_angles_cls.append(angle_cls)
                dec_out_angles.append(angle)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox


        if return_last_query:
            return (paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits),
                    paddle.stack(dec_out_angles_cls), paddle.stack(dec_out_angles),
                    output)
        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits), paddle.stack(dec_out_angles_cls), paddle.stack(dec_out_angles)
class TransformerDecoder_RadarCamera_Rotate(nn.Layer):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 angle_max,
                 angle_proj,
                 half_pi_bin,
                 eval_idx=-1):
        super(TransformerDecoder_RadarCamera_Rotate, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.angle_max = angle_max
        self.angle_proj = angle_proj
        self.half_pi_bin = half_pi_bin

    def forward(self,
                gt_meta,
                tgt_radar,
                ref_points_unact,
                ref_angle_cls,
                ref_angle,
                radar_memory,
                camera_memory,
                radar_memory_spatial_shapes,
                radar_memory_level_start_index,
                camera_memory_spatial_shapes,
                camera_memory_level_start_index,
                bbox_head,
                score_head,
                angle_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score=None,
                mask_vis=None):
        output = tgt_radar
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_angles_cls = []
        dec_out_angles = []

        ref_points_detach = F.sigmoid(ref_points_unact)
        angle = ref_angle

        gt_meta['layer'] = 0
        b, l = ref_points_detach.shape[:2]

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                gt_meta, output, ref_points_input, angle, radar_memory,
                camera_memory, radar_memory_spatial_shapes,
                radar_memory_level_start_index, camera_memory_spatial_shapes,
                camera_memory_level_start_index, self.angle_max,
                self.half_pi_bin, attn_mask, memory_mask, query_pos_embed,
                topk_ind_mask, topk_score, mask_vis)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            angle_cls = angle_head[i](output)
            angle = F.softmax(
                angle_cls.reshape([b, l, 1, self.angle_max + 1]),
                axis=-1).matmul(self.angle_proj)

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_angles_cls.append(angle_cls)
                dec_out_angles.append(angle)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(
                            bbox_head[i](output) +
                            inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_angles_cls.append(angle_cls)
                dec_out_angles.append(angle)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return (paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits),
                paddle.stack(dec_out_angles_cls), paddle.stack(dec_out_angles))

class TransformerDecoder_RANK(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_RANK, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                rank_aware_content_query,
                rank_adaptive_classhead_emb,
                pre_racq_trans,
                post_racq_trans,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            #rank query layer
            if i>=1:
                output_target = paddle.gather_nd(output[:,-self.num_queries:,:], rank_indices)
                output_dn = output[:,:all_count-self.num_queries,:]
                # output = paddle.concat([output_target,output_dn], axis=1)
                concat_term = pre_racq_trans[i - 1](
                    paddle.expand(paddle.unsqueeze(rank_aware_content_query[i - 1].weight, axis=0),[bs,-1,-1])
                )
                output_target = paddle.concat([output_target,concat_term], axis=2)
                output_target = post_racq_trans[i - 1](output_target)
                output = paddle.concat([output_dn,output_target], axis=1)


            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))  #inter_ref_bbox is new reference point ; ref_points_detach is reference point

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])

            #generate rank indicates
            outputs_class_tmp = score_head[i](output)
            sigmoid_outputs = paddle.nn.functional.sigmoid(outputs_class_tmp)
            rank_basis = paddle.max(sigmoid_outputs, axis=2, keepdim=False)
            rank_indices = paddle.argsort(rank_basis[:, -self.num_queries:], descending=True, axis=1)
            rank_indices = rank_indices.detach()


            # rank the reference points
            batch_ind = paddle.arange(end=bs, dtype=rank_indices.dtype)
            batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
            rank_indices = paddle.stack([batch_ind, rank_indices], axis=-1)
            inter_ref_bbox_target = paddle.gather_nd(inter_ref_bbox[:,-self.num_queries:,:], rank_indices)

            if self.training:
                inter_ref_bbox_dn = inter_ref_bbox[:,:all_count-self.num_queries,:]
                inter_ref_bbox = paddle.concat([inter_ref_bbox_dn,inter_ref_bbox_target], axis=1)
            else:
                inter_ref_bbox = inter_ref_bbox_target

            if self.training:
                #dec_out_logits.append(score_head[i](output))
                rank_adaptive_classhead_emb_lvl = paddle.unsqueeze(rank_adaptive_classhead_emb[i].weight, axis=0).tile(
                    repeat_times=[bs, 1, 1])
                output_class = score_head[i](output)
                output_class_target = paddle.gather_nd(output_class[:,-self.num_queries:,:] + rank_adaptive_classhead_emb_lvl, rank_indices)
                    # output_class[:,-self.num_queries:,:] + rank_adaptive_classhead_emb_lvl
                output_class_dn = output_class[:,:all_count-self.num_queries,:]
                output_class = paddle.concat([output_class_dn,output_class_target], axis=1)
                dec_out_logits.append(output_class)

                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    # dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                    #     ref_points_detach)))
                else:
                    inter_ref_bbox_detach = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points))
                    inter_ref_bbox_detach_target = paddle.gather_nd(inter_ref_bbox_detach[:,-self.num_queries:,:], rank_indices)
                    inter_ref_bbox_detach_dn = inter_ref_bbox_detach[:, :all_count - self.num_queries, :]
                    inter_ref_bbox_detach = paddle.concat([inter_ref_bbox_detach_dn, inter_ref_bbox_detach_target], axis=1)
                    dec_out_bboxes.append(inter_ref_bbox_detach)

            elif i == self.eval_idx:
                # dec_out_logits.append(score_head[i](output))
                rank_adaptive_classhead_emb_lvl = paddle.unsqueeze(rank_adaptive_classhead_emb[i].weight, axis=0).tile(
                    repeat_times=[bs, 1, 1])
                output_class = paddle.gather_nd(score_head[i](output) + rank_adaptive_classhead_emb_lvl,rank_indices)
                dec_out_logits.append(output_class)

                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_Group(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Group, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)


        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])






            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_Groupx3_Missing(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx3_Missing, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                query_complementary_for_vis=None,
                query_complementary_for_ir=None,
                pre_racq_trans=None,
                post_racq_trans=None,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)


            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            # if self.training or eval_all:
            # #complementary for vis and ir
            #     #vis
            #     output_vis = output[:,-self.num_queries * 2 : -self.num_queries, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_vis[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_vis = paddle.concat([output_vis, concat_term], axis=2)
            #     output_vis = post_racq_trans[i](output_vis)
            #         #ir
            #     output_ir = output[:, -self.num_queries:, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_ir[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_ir = paddle.concat([output_ir, concat_term], axis=2)
            #     output_ir = post_racq_trans[i](output_ir)
            #
            #     output = paddle.concat([output[:,:-self.num_queries * 2,:], output_vis, output_ir], axis=1)


            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)

class TransformerDecoder_Groupx3_Missing_V2(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx3_Missing_V2, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                # vis_memory,
                # ir_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                # query_complementary_for_vis,
                # query_complementary_for_ir,
                # pre_racq_trans,
                # post_racq_trans,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)


            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            # if self.training or eval_all:
            # #complementary for vis and ir
            #     #vis
            #     output_vis = output[:,-self.num_queries * 2 : -self.num_queries, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_vis[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_vis = paddle.concat([output_vis, concat_term], axis=2)
            #     output_vis = post_racq_trans[i](output_vis)
            #         #ir
            #     output_ir = output[:, -self.num_queries:, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_ir[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_ir = paddle.concat([output_ir, concat_term], axis=2)
            #     output_ir = post_racq_trans[i](output_ir)
            #
            #     output = paddle.concat([output[:,:-self.num_queries * 2,:], output_vis, output_ir], axis=1)


            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_Groupx3_Missing_V3(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx3_Missing_V3, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                vis_memory,
                ir_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                # query_complementary_for_vis,
                # query_complementary_for_ir,
                # pre_racq_trans,
                # post_racq_trans,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)


            output = layer(gt_meta, output, ref_points_input, memory,vis_memory,ir_memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            # if self.training or eval_all:
            # #complementary for vis and ir
            #     #vis
            #     output_vis = output[:,-self.num_queries * 2 : -self.num_queries, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_vis[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_vis = paddle.concat([output_vis, concat_term], axis=2)
            #     output_vis = post_racq_trans[i](output_vis)
            #         #ir
            #     output_ir = output[:, -self.num_queries:, :]
            #     concat_term = pre_racq_trans[i](
            #         paddle.expand(paddle.unsqueeze(query_complementary_for_ir[i].weight, axis=0), [bs, -1, -1])
            #     )
            #     output_ir = paddle.concat([output_ir, concat_term], axis=2)
            #     output_ir = post_racq_trans[i](output_ir)
            #
            #     output = paddle.concat([output[:,:-self.num_queries * 2,:], output_vis, output_ir], axis=1)


            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)

class TransformerDecoder_Groupx5_Missing(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx5_Missing, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                query_complementary_for_vis,
                query_complementary_for_ir,
                pre_racq_trans,
                post_racq_trans,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)


            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            if self.training or eval_all:
            #complementary for vis and ir
                #vis
                output_vis = output[:,-self.num_queries * 2 : -self.num_queries, :]
                concat_term = pre_racq_trans[i](
                    paddle.expand(paddle.unsqueeze(query_complementary_for_vis[i].weight, axis=0), [bs, -1, -1])
                )
                output_vis = paddle.concat([output_vis, concat_term], axis=2)
                output_vis = post_racq_trans[i](output_vis)
                    #ir
                output_ir = output[:, -self.num_queries:, :]
                concat_term = pre_racq_trans[i](
                    paddle.expand(paddle.unsqueeze(query_complementary_for_ir[i].weight, axis=0), [bs, -1, -1])
                )
                output_ir = paddle.concat([output_ir, concat_term], axis=2)
                output_ir = post_racq_trans[i](output_ir)

                output = paddle.concat([output[:,:-self.num_queries * 2,:], output_vis, output_ir], axis=1)


            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out.append(output)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox
        if dec_out != []:
            return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits), paddle.stack(dec_out)
        else:
            return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits), None


class TransformerDecoder_Groupx4(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx4, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)


        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)


            # # refenece point visi
            # topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # color_p = (128,0,128)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 3:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_p,-1)
            #                     cv2.circle(ir_imgs[xx], (
            #                     round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius + 2, color_p, -1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_g, -1)
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]),
            #                                 round(real_reference_point[xx][ii][1])),
            #                                radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/'+'init_reference_point_level_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/'+'init_reference_point_level_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_Groupx4_RANK(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_Groupx4_RANK, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.num_queries = 300

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                rank_aware_content_query,
                rank_adaptive_classhead_emb,
                pre_racq_trans,
                post_racq_trans,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None,
                eval_all = False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        bs,all_count,_ = tgt.shape

        gt_meta['layer'] = 0


        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            # rank query layer
            if i >= 1:
                if self.training:
                    output_target = paddle.gather_nd(output[:, -self.num_queries * 4:, :], rank_indices)
                    output_dn = output[:, :all_count - self.num_queries * 4, :]
                    # output = paddle.concat([output_target,output_dn], axis=1)
                    concat_term = pre_racq_trans[i - 1](
                        paddle.expand(paddle.unsqueeze(rank_aware_content_query[i - 1].weight, axis=0), [bs, -1, -1])
                    )
                    output_target = paddle.concat([output_target, concat_term], axis=2)
                    output_target = post_racq_trans[i - 1](output_target)
                    output = paddle.concat([output_dn, output_target], axis=1)
                else:
                    output_target = paddle.gather_nd(output, rank_indices)
                    #output_dn = output[:, :all_count - self.num_queries * 4, :]
                    # output = paddle.concat([output_target,output_dn], axis=1)
                    concat_term = pre_racq_trans[i - 1](
                        paddle.expand(paddle.unsqueeze(rank_aware_content_query[i - 1].weight[:self.num_queries,:], axis=0), [bs, -1, -1])
                    )
                    output_target = paddle.concat([output_target, concat_term], axis=2)
                    output_target = post_racq_trans[i - 1](output_target)
                    #output = paddle.concat([output_dn, output_target], axis=1)
                    output = output_target

            output = layer(gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,eval_all)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            #visul gropu bbox
            # real_hw = [1024, 768]
            # real_hw = np.array(real_hw)
            # bboxes1 = np.array(inter_ref_bbox)[0,:300,:2] * real_hw
            # bboxes2 = np.array(inter_ref_bbox)[0,300:600,:2] * real_hw
            # bboxes3 = np.array(inter_ref_bbox)[0,600:, :2] * real_hw
            # image = np.ones((768,1024,3), dtype=np.uint8) * 255
            # for mm in range(300):
            #     cv2.circle(image,(round(bboxes1[mm][0]),round(bboxes1[mm][1])),radius=1,color=(255,0,0),thickness=-1)
            #     cv2.circle(image, (round(bboxes2[mm][0]), round(bboxes2[mm][1])),radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(image, (round(bboxes3[mm][0]), round(bboxes3[mm][1])),radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/log/test.jpg',image)


            # # refenece point visi
            # topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # color_p = (128,0,128)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 3:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_p,-1)
            #                     cv2.circle(ir_imgs[xx], (
            #                     round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius + 2, color_p, -1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_g, -1)
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]),
            #                                 round(real_reference_point[xx][ii][1])),
            #                                radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/'+'init_reference_point_level_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/'+'init_reference_point_level_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])

            # generate rank indicates
            outputs_class_tmp = score_head[i](output)
            sigmoid_outputs = paddle.nn.functional.sigmoid(outputs_class_tmp)
            rank_basis = paddle.max(sigmoid_outputs, axis=2, keepdim=False)
            if self.training:
                rank_indices_4 = paddle.argsort(rank_basis[:, -self.num_queries:], descending=True, axis=1)
                rank_indices_3 = paddle.argsort(rank_basis[:, -self.num_queries * 2:-self.num_queries], descending=True, axis=1)
                rank_indices_2 = paddle.argsort(rank_basis[:, -self.num_queries * 3:-self.num_queries * 2], descending=True,
                                                axis=1)
                rank_indices_1 = paddle.argsort(rank_basis[:, -self.num_queries * 4:-self.num_queries * 3], descending=True,
                                                axis=1)
                rank_indices = paddle.concat([rank_indices_1,rank_indices_2 + paddle.ones_like(rank_indices_1) * self.num_queries,
                                              rank_indices_3 + paddle.ones_like(rank_indices_1) * self.num_queries * 2,
                                              rank_indices_4 + paddle.ones_like(rank_indices_1) * self.num_queries * 3
                                              ], axis=1)
            else:
                rank_indices = paddle.argsort(rank_basis, descending=True,
                                                axis=1)
            rank_indices = rank_indices.detach()

            # rank the reference points
            batch_ind = paddle.arange(end=bs, dtype=rank_indices.dtype)
            if self.training:
                batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries * 4])
                rank_indices = paddle.stack([batch_ind, rank_indices], axis=-1)
                inter_ref_bbox_target = paddle.gather_nd(inter_ref_bbox[:, -self.num_queries * 4:, :], rank_indices)
            else:
                batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
                rank_indices = paddle.stack([batch_ind, rank_indices], axis=-1)
                inter_ref_bbox_target = paddle.gather_nd(inter_ref_bbox, rank_indices)

            if self.training:
                inter_ref_bbox_dn = inter_ref_bbox[:, :all_count - self.num_queries * 4, :]
                inter_ref_bbox = paddle.concat([inter_ref_bbox_dn, inter_ref_bbox_target], axis=1)
            else:
                inter_ref_bbox = inter_ref_bbox_target

            if self.training:
                # dec_out_logits.append(score_head[i](output))
                rank_adaptive_classhead_emb_lvl = paddle.unsqueeze(rank_adaptive_classhead_emb[i].weight, axis=0).tile(
                    repeat_times=[bs, 1, 1])
                output_class = score_head[i](output)
                output_class_target = paddle.gather_nd(
                    output_class[:, -self.num_queries * 4:, :] + rank_adaptive_classhead_emb_lvl, rank_indices)
                # output_class[:,-self.num_queries:,:] + rank_adaptive_classhead_emb_lvl
                output_class_dn = output_class[:, :all_count - self.num_queries * 4, :]
                output_class = paddle.concat([output_class_dn, output_class_target], axis=1)
                dec_out_logits.append(output_class)

                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    # dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                    #     ref_points_detach)))
                else:
                    inter_ref_bbox_detach = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                        ref_points))
                    inter_ref_bbox_detach_target = paddle.gather_nd(inter_ref_bbox_detach[:, -self.num_queries * 4:, :],
                                                                    rank_indices)
                    inter_ref_bbox_detach_dn = inter_ref_bbox_detach[:, :all_count - self.num_queries * 4, :]
                    inter_ref_bbox_detach = paddle.concat([inter_ref_bbox_detach_dn, inter_ref_bbox_detach_target],
                                                          axis=1)
                    dec_out_bboxes.append(inter_ref_bbox_detach)

            elif i == self.eval_idx:
                # dec_out_logits.append(score_head[i](output))
                rank_adaptive_classhead_emb_lvl = paddle.unsqueeze(rank_adaptive_classhead_emb[i].weight[:self.num_queries,:], axis=0).tile(
                    repeat_times=[bs, 1, 1])
                output_class = paddle.gather_nd(score_head[i](output) + rank_adaptive_classhead_emb_lvl, rank_indices)
                dec_out_logits.append(output_class)

                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


class TransformerDecoder_BA(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_BA, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        self.mask_transform = paddle.create_parameter(shape=[hidden_dim,hidden_dim],
                                            dtype='float32',
                                            name='mask_transform',
                                            is_bias=False,
                                            )

    def forward(self,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                enc_topk_bboxes,
                bbox_head,
                score_head,
                query_pos_head,
                value_BA_head=None,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score = None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        gt_meta['layer'] = 0
        bs = len(gt_meta['im_id'])



        # visual code
        # vis_imgs = []
        # ir_imgs = []
        # hh = []
        # ww = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        #     #h[xx],w[xx],_ = vis_imgs[xx].shape
        #     hh.append(vis_imgs[xx].shape[0])
        #     ww.append(vis_imgs[xx].shape[1])





        for i, layer in enumerate(self.layers):
            #transfer_memory = memory


            #memory bbox attention efficent
            attention_300box = ref_points_detach[:,-300:,:].detach()
            attention_300score = topk_score[:,-300:].detach()
            sort_index = paddle.argsort(attention_300score,axis=1,descending=True)
            batch_ind = paddle.arange(end=bs, dtype=sort_index.dtype)
            batch_ind = batch_ind.unsqueeze(-1).tile([1, 300])
            sort_index = paddle.stack([batch_ind, sort_index], axis=-1)

            sort_attention_300box = paddle.gather_nd(attention_300box,sort_index)  # unsigmoided.
            sort_topk_ind_mask = paddle.gather_nd(topk_ind_mask,sort_index)

            #visul topk30 score
            #sort_topk_50score = paddle.gather_nd(attention_300score,sort_index)[:,:30]

            sort_attention_50box = sort_attention_300box[:,:30,:]
            sort_topk_ind_50mask = sort_topk_ind_mask[:,:30]

            #box condition ellegal
            condition = (sort_attention_50box[:,:,0] - sort_attention_50box[:,:,2] / 2 >0 ) & (sort_attention_50box[:,:,1] - sort_attention_50box[:,:,3] /2 >0)
            #bbox_legal_mask = paddle.cast(condition,dtype='float64')
            sort_topk_ind_50mask = paddle.where(condition, sort_topk_ind_50mask, paddle.zeros_like(sort_topk_ind_50mask))

                #vis box
            mmask_vis = sort_topk_ind_50mask == 1

                #ir box
            mmask_ir = sort_topk_ind_50mask == 2

            all_mask_visir = []
            #tic_1 = time.time()
            for n in range(bs):

                indx_vis = paddle.nonzero(mmask_vis[n]).flatten()
                indx_ir = paddle.nonzero(mmask_ir[n]).flatten()
                all_mask_vis = paddle.zeros(shape=(0,),dtype=paddle.int32)
                all_mask_ir = paddle.zeros(shape=(0,),dtype=paddle.int32)


                # #visual code
                # cv2.imwrite(
                #     '/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-BA-yx-newm3fd/bbox-attention-mask/layer'+str(i)+'/' +
                #     gt_meta['vis_im_file'][n].split('/')[-1].split('.')[0] + '_vis.png',
                #     vis_imgs[n])
                #
                # cv2.imwrite(
                #     '/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-BA-yx-newm3fd/bbox-attention-mask/layer' + str(
                #         i) + '/' +
                #     gt_meta['ir_im_file'][n].split('/')[-1].split('.')[0] + '_ir.png',
                #     ir_imgs[n])

                for xx in range(3):
                    #_bx = vis_b * paddle.to_tensor(memory_spatial_shapes[xx]+memory_spatial_shapes[xx])
                    all_bx = sort_attention_50box[n] * paddle.to_tensor(memory_spatial_shapes[xx]+memory_spatial_shapes[xx])
                    # x = all_bx[:,0]
                    # y = all_bx[:,1]
                    # w = all_bx[:,2]
                    # h = all_bx[:,3]
                    # x = x - w /2
                    # y = y - h /2

                    x = np.array(all_bx[:, 0])
                    y = np.array(all_bx[:, 1])
                    w = np.array(all_bx[:, 2])
                    h = np.array(all_bx[:, 3])
                    x = x - w / 2
                    y = y - h / 2

                    H = memory_spatial_shapes[xx][0]
                    W = memory_spatial_shapes[xx][1]
                    mask_vis = np.zeros([H,W])
                    mask_ir = np.zeros([H,W])

                    # start_x = paddle.cast(paddle.round(x), dtype='int32')
                    # end_x = paddle.cast(paddle.round(x + w), dtype='int32')
                    # start_y = paddle.cast(paddle.round(y), dtype='int32')
                    # end_y = paddle.cast(paddle.round(y + h), dtype='int32')
                    start_x = np.round(x).astype('int32')
                    end_x = np.round(x+w).astype('int32')
                    start_y = np.round(y).astype('int32')
                    end_y = np.round(y+h).astype('int32')


                    #tic11 = time.time()
                    #

                    for iindx in indx_vis:
                        # if x[iindx] > 0 and y[iindx] > 0:
                        #     None
                            #mask_vis[paddle.round(x[iindx]):paddle.round(x[iindx]+w[iindx]),paddle.round(y[iindx]):paddle.round(y[iindx]+h[iindx])] = 1

                        #mask_vis[start_x[iindx]:end_x[iindx], start_y[iindx]:end_y[iindx]] = 1

                        mask_vis[start_y[iindx]:end_y[iindx]+1,start_x[iindx]:end_x[iindx]+1] = 1



                        #print(start_x[iindx],end_x[iindx], start_y[iindx],end_y[iindx])
                        #print(mask_vis[start_x[iindx]:end_x[iindx], start_y[iindx]:end_y[iindx]],paddle.full([end_x[iindx] - start_x[iindx], end_y[iindx] - start_y[iindx]], 1))
                        #mask_vis[start_x[iindx]:end_x[iindx], start_y[iindx]:end_y[iindx]] = paddle.full([end_x[iindx] - start_x[iindx], end_y[iindx] - start_y[iindx]], 1)
                        #mask_vis[5:10, 7:15] = 1

                    #visual code

                    # cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-BA-yx-newm3fd/bbox-attention-mask/layer'+str(i)+'/'+gt_meta['vis_im_file'][n].split('/')[-1].split('.')[0]+'_vis_mask'+str(xx)+'.png',
                    #             cv2.resize(np.array(mask_vis) * 255,(ww[n],hh[n])))


                    mask_vis = paddle.flatten(paddle.to_tensor(mask_vis,dtype='float32'))
                    all_mask_vis = paddle.concat([all_mask_vis,mask_vis],axis=-1)

                    for irndx in indx_ir:
                        # if x[irndx] > 0 and y[irndx] > 0:
                        #     None
                            # mask_ir[paddle.round(x[irndx]):paddle.round(x[irndx] + w[irndx]),
                            # paddle.round(y[irndx]):paddle.round(y[irndx] + h[irndx])] = 1
                        #mask_ir[start_x[irndx]:end_x[irndx], start_y[irndx]:end_y[irndx]] = 1
                        mask_ir[start_y[irndx]:end_y[irndx]+1,start_x[irndx]:end_x[irndx]+1] = 1

                        #mask_ir[start_x[irndx]:end_x[irndx], start_y[irndx]:end_y[irndx]] = paddle.full([end_x[irndx] - start_x[irndx], end_y[irndx] - start_y[irndx]], 1)

                        #mask_ir[5:10, 7:15] = 1

                    # cv2.imwrite(
                    #     '/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-BA-yx-newm3fd/bbox-attention-mask/layer' + str(
                    #         i) + '/' + gt_meta['vis_im_file'][n].split('/')[-1].split('.')[0] + '_ir_mask' + str(
                    #         xx) + '.png',
                    #     cv2.resize(np.array(mask_ir) * 255, (ww[n], hh[n])))

                    mask_ir = paddle.flatten(paddle.to_tensor(mask_ir,dtype='float32'))
                    all_mask_ir = paddle.concat([all_mask_ir,mask_ir],axis=-1)

                    #tic22 = time.time()
                    #print(tic22 - tic11)

                all_mask_visir.append( paddle.concat([all_mask_vis,all_mask_ir],axis=-1).unsqueeze(0) )

            all_mask = paddle.concat(all_mask_visir,axis=0).unsqueeze(-1)

            #chanal select
            all_mask = paddle.tile(all_mask,[1,1,256])
            mask_code = paddle.matmul(all_mask, self.mask_transform)
            transfer_memory = memory + mask_code

            #chanal select-MLP
            # mask_code = value_BA_head(all_mask)
            # transfer_memory = memory + mask_code
            #memory select
            # memory_select = memory * all_mask
            # mask_code = paddle.matmul(memory_select, self.mask_transform)
            # transfer_memory = memory + mask_code



                    #mmask = (cols_expanded >=x.unsqueeze(0)) & (cols_expanded < (x+w).unsqueeze(0)) & (rows_expanded >= y.unsqueeze(0)) & (rows_expanded < (y+h).unsqueeze(0))


            # #memory attention use query
            # ref_query_box_detach = enc_topk_bboxes.detach()
            # ref_query_box_detach = ref_query_box_detach[:,:50,:]
            # topk_score = topk_score[:,-300:-200].detach()
            # topk_ind_mask = topk_ind_mask[:,:100]
            #
            # memory_vis1 = paddle.zeros((len(gt_meta['im_id']),memory_spatial_shapes[0][0],memory_spatial_shapes[0][1]),dtype='float32')
            # memory_vis2 = paddle.zeros(
            #     (len(gt_meta['im_id']), memory_spatial_shapes[1][0], memory_spatial_shapes[1][1]), dtype='float32')
            # memory_vis3 = paddle.zeros(
            #     (len(gt_meta['im_id']), memory_spatial_shapes[2][0], memory_spatial_shapes[2][1])
            # )
            # memory_ir1 = paddle.zeros(
            #     (len(gt_meta['im_id']), memory_spatial_shapes[3][0], memory_spatial_shapes[3][1]), dtype='float32')
            # memory_ir2 = paddle.zeros(
            #     (len(gt_meta['im_id']), memory_spatial_shapes[4][0], memory_spatial_shapes[4][1]), dtype='float32')
            # memory_ir3_mask = paddle.zeros(
            #     (len(gt_meta['im_id']), memory_spatial_shapes[5][0] * memory_spatial_shapes[5][1]), dtype='float32')
            #
            #     #vis box
            # mmask_vis = paddle.logical_and(topk_score > self.alfa[i],topk_ind_mask == 1)
            # expand_mmask = paddle.unsqueeze(mmask_vis, axis=-1)
            # expand_mmask = paddle.cast(expand_mmask,dtype='float32')
            # expand_mmask = paddle.expand(expand_mmask, shape=[-1,-1,4])
            # vis_b = paddle.multiply(ref_query_box_detach,expand_mmask)
            # vis_b1 = vis_b * paddle.to_tensor(memory_spatial_shapes[0]+memory_spatial_shapes[0])
            # vis_b2 = vis_b * paddle.to_tensor(memory_spatial_shapes[1]+memory_spatial_shapes[1])
            # #vis_b = paddle.gather_nd(ref_query_box_detach,iidex[:,1])
            #     #ir box
            # mmask_ir = paddle.logical_and(topk_score > self.alfa[i], topk_ind_mask == 2)
            # expand_mmask = paddle.unsqueeze(mmask_ir, axis=-1)
            # expand_mmask = paddle.cast(expand_mmask, dtype='float32')
            # expand_mmask = paddle.expand(expand_mmask, shape=[-1, -1, 4])
            # ir_b = paddle.multiply(ref_query_box_detach, expand_mmask)
            # ir_b1 = ir_b * paddle.to_tensor(memory_spatial_shapes[2] + memory_spatial_shapes[2])
            # ir_b2 = ir_b * paddle.to_tensor(memory_spatial_shapes[3] + memory_spatial_shapes[3])
            #
            # for ii in range(len(gt_meta['im_id'])):
            #     indx_vis = paddle.fluid.layers.where(mmask_vis[ii] == True)
            #     indx_ir = paddle.fluid.layers.where(mmask_ir[ii] == True)
            #
            #     for iidx in indx_vis:
            #         if vis_b1[ii][iidx][0]<(vis_b1[ii][iidx][2]/2) or vis_b1[ii][iidx][1]<(vis_b1[ii][iidx][3]/2):
            #             continue
            #         memory_vis1[int(paddle.round(vis_b1[ii][iidx][0]-vis_b1[ii][iidx][2]/2)):int(paddle.round(vis_b1[ii][iidx][0]+vis_b1[ii][iidx][2]/2))+1,
            #         int(paddle.round(vis_b1[ii][iidx][1]-vis_b1[ii][iidx][3]/2)):int(paddle.round(vis_b1[ii][iidx][1]+vis_b1[ii][iidx][3]/2))+1] = 1
            #         if vis_b2[ii][iidx][0]<(vis_b2[ii][iidx][2]/2) or vis_b2[ii][iidx][1]<(vis_b2[ii][iidx][3]/2):
            #             continue
            #         memory_vis2[int(paddle.round(vis_b2[ii][iidx][0] - vis_b2[ii][iidx][2] / 2)):int(paddle.round(
            #             vis_b2[ii][iidx][0] + vis_b2[ii][iidx][2] / 2))+1,
            #         int(paddle.round(vis_b2[ii][iidx][1] - vis_b2[ii][iidx][3] / 2)):int(paddle.round(
            #             vis_b2[ii][iidx][1] + vis_b2[ii][iidx][3] / 2))+1] = 1
            #     for iidx in indx_ir:
            #         if ir_b1[ii][iidx][0]<(ir_b1[ii][iidx][2]/2) or ir_b1[ii][iidx][1]<(ir_b1[ii][iidx][3]/2):
            #             continue
            #         memory_ir1[int(paddle.round(ir_b1[ii][iidx][0] - ir_b1[ii][iidx][2] / 2)):int(paddle.round(
            #             ir_b1[ii][iidx][0] + ir_b1[ii][iidx][2] / 2)+1),
            #         int(paddle.round(ir_b1[ii][iidx][1] - ir_b1[ii][iidx][3] / 2)):int(paddle.round(
            #             ir_b1[ii][iidx][1] + ir_b1[ii][iidx][3] / 2))+1] = 1
            #         if ir_b2[ii][iidx][0]<(ir_b2[ii][iidx][2]/2) or ir_b2[ii][iidx][1]<(ir_b2[ii][iidx][3]/2):
            #             continue
            #         memory_ir2[int(paddle.round(ir_b2[ii][iidx][0] - ir_b2[ii][iidx][2] / 2)):int(paddle.round(
            #             ir_b2[ii][iidx][0] + ir_b2[ii][iidx][2] / 2))+1,
            #         int(paddle.round(ir_b2[ii][iidx][1] - ir_b2[ii][iidx][3] / 2)):int(paddle.round(
            #             ir_b2[ii][iidx][1] + ir_b2[ii][iidx][3] / 2))+1] = 1
            #
            # memory_vis1_mask = paddle.reshape(memory_vis1,shape=[4, -1])
            # memory_vis2_mask = paddle.reshape(memory_vis2,shape=[4, -1])
            # memory_ir1_mask = paddle.reshape(memory_ir1,shape=[4,-1])
            # memory_ir2_mask = paddle.reshape(memory_ir2,shape=[4,-1])
            # mymemory_mask = paddle.concat([memory_vis1_mask,memory_vis2_mask,memory_ir1_mask,memory_ir2_mask,memory_irvis3_mask],axis=1)
            # mymemory_mask = paddle.unsqueeze(mymemory_mask,axis=-1)
            # memory_slect = memory * mymemory_mask
            # memory_slect = memory_slect @ self.mask_transform
            #
            # memory= memory + memory_slect
            #############

            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(gt_meta, output, ref_points_input, transfer_memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score,sort_index)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])




            topk_score_all = score_head[i](output)
            topk_score = paddle.max(topk_score_all,axis=-1).detach()

            if self.training:
                dec_out_logits.append(topk_score_all)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(topk_score_all)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)



class TransformerDecoder_split(nn.Layer):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder_split, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx


    def forward(self,
                num_queries,
                gt_meta,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                enc_topk_bboxes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                topk_ind_mask=None,
                topk_score=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        gt_meta['layer'] = 0
        bs = len(gt_meta['im_id'])


        #split vis and ir query


        # 使用逻辑索引选择mask为1的位置的tensor值
        # selected_indices = paddle.nonzero(topk_ind_mask == 1)  # 获取mask为1的位置的索引
        # vis_tgt = paddle.gather_nd(visir_tgt, selected_indices)  # 根据索引选择对应位置的值
        # vis_tgt = paddle.reshape(vis_tgt, [4, -1, 256])  # 调整形状为[4, n, 256]

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(num_queries,gt_meta, output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,topk_ind_mask,topk_score)

            gt_meta['layer'] = gt_meta['layer'] + 1

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach))

            # # refenece point visi
            #topk_score = np.array(paddle.max(score_head[i](output),axis=-1))
            # enc_topk_bboxes = np.array(inter_ref_bbox)
            #
            # #   ##plot point
            # bs = len(gt_meta['im_id'])
            # vis_imgs = []
            # ir_imgs = []
            # for xx in range(bs):
            #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
            #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
            # h,w,_ = vis_imgs[0].shape
            # real_hw = [w,h]
            # real_hw = np.array(real_hw)
            # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
            #
            # radius = 4
            # color_r = (0,0,255)
            # color_b = (230,216,173)
            # color_g = (152,251, 152)
            # if i==5:
            #     for xx in range(bs):
            #         for ii in range(300):
            #             if topk_ind_mask[xx][ii] == 1:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(vis_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             elif topk_ind_mask[xx][ii] == 2:
            #                 if topk_score[xx][ii] > 0:
            #                     cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
            #                                radius+2,color_r,-1)
            #                 else:
            #                     cv2.circle(ir_imgs[xx],
            #                                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                                radius-2, color_b, -1)
            #             else:
            #                 cv2.circle(vis_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #                 cv2.circle(ir_imgs[xx],
            #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
            #                            radius - 2, color_g, -1)
            #
            #     for ii in range(bs):
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
            #         cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/'+'mask_ir_point_'+str(i+1)+'/'+
            #                     gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])






            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                            ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)


@register
class RTDETRTransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask=topk_ind_mask,
            topk_score=topk_score)

        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits


@register
class Multi_RTDETRTransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)
        # for l in self.input_proj:
        #     xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        visir_feats = vis_feats + ir_feats
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        memory = paddle.split(visir_memory, 2, axis=1)[0] + paddle.split(visir_memory, 2, axis=1)[1]
        spatial_shapes = visir_spatial_shapes[0:3]

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits


@register
class Multi_RTDETRTransformerv2_6lselect_3LEVEL(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformerv2_6lselect_3LEVEL, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        visir_feats = vis_feats + ir_feats
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        memory = paddle.split(visir_memory, 2, axis=1)[0] + paddle.split(visir_memory, 2, axis=1)[1]
        spatial_shapes = visir_spatial_shapes[0:3]
        level_start_index = visir_level_start_index[0:3]

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            visir_memory, visir_spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits



@register
class Multi_RTDETRTransformer_V2_3LEVEL(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 # backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 # visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 # num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer_V2_3LEVEL, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        #self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        #self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        #self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    # def _build_visir_input_proj_layer(self, backbone_feat_channels):
    #     self.input_proj_visir = nn.LayerList()
    #     for in_channels in backbone_feat_channels:
    #         self.input_proj_visir.append(
    #             nn.Sequential(
    #                 ('conv', nn.Conv2D(
    #                     in_channels,
    #                     self.hidden_dim,
    #                     kernel_size=1,
    #                     bias_attr=False)), ('norm', nn.BatchNorm2D(
    #                         self.hidden_dim,
    #                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
    #                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
    #     in_channels = backbone_feat_channels[-1]
    #     for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
    #         self.input_proj_visir.append(
    #             nn.Sequential(
    #                 ('conv', nn.Conv2D(
    #                     in_channels,
    #                     self.hidden_dim,
    #                     kernel_size=3,
    #                     stride=2,
    #                     padding=1,
    #                     bias_attr=False)), ('norm', nn.BatchNorm2D(
    #                         self.hidden_dim,
    #                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
    #                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
    #         in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    # def _get_encoder_visir_input(self, feats):
    #     # get projection features
    #     proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
    #     if self.num_visir_levels > len(proj_feats):
    #         len_srcs = len(proj_feats)
    #         for i in range(len_srcs, self.num_visir_levels):
    #             if i == len_srcs:
    #                 proj_feats.append(self.input_proj_visir[i](feats[-1]))
    #             else:
    #                 proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))
    #
    #     # get encoder inputs
    #     feat_flatten = []
    #     spatial_shapes = []
    #     level_start_index = [0, ]
    #     for i, feat in enumerate(proj_feats):
    #         _, _, h, w = feat.shape
    #         # [b, c, h, w] -> [b, h*w, c]
    #         feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
    #         # [num_levels, 2]
    #         spatial_shapes.append([h, w])
    #         # [l], start index of each level
    #         level_start_index.append(h * w + level_start_index[-1])
    #
    #     # [b, l, c]
    #     feat_flatten = paddle.concat(feat_flatten, 1)
    #     level_start_index.pop()
    #     return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits

@register
class Multi_RTDETRTransformer_V4(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 # visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 # num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer_V4, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        # self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        # self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        #self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        ##########
        # decoder_layer = TransformerDecoderLayer(
        #     hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
        #     num_decoder_points)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        ############

        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        #visir_feats = vis_feats + ir_feats
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        # (visir_memory, visir_spatial_shapes,
        #  visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask=None)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits


@register
class Multi_Group_RTDETRTransformer_V3(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False):
        super(Multi_Group_RTDETRTransformer_V3, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Group(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_Group(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class_visir = self.enc_score_head_visir(output_memory)
        enc_outputs_coord_unact_visir = self.enc_bbox_head_visir(output_memory) + anchors

        if self.training or self.eval_all is True:
            output_memory_vis = paddle.split(output_memory, 2, axis=1)[0]
            output_memory_ir = paddle.split(output_memory, 2, axis=1)[1]
            anchors_vis = paddle.split(anchors, 2, axis=1)[0]
            anchors_ir = paddle.split(anchors, 2, axis=1)[1]

            enc_outputs_class_vis = self.enc_score_head_vis(output_memory_vis)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory_vis) + anchors_vis

            enc_outputs_class_ir = self.enc_score_head_ir(output_memory_ir)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory_ir) + anchors_ir

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1), self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class_visir.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        # topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        # topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact_visir,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class_visir, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_memory_vis, topk_ind_vis)
                target_ir = paddle.gather_nd(output_memory_ir, topk_ind_ir)
                target = paddle.concat([target,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score


@register
class Multi_Groupx3_RTDETRTransformer_V3_Missing(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False):
        super(Multi_Groupx3_RTDETRTransformer_V3_Missing, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx3_Missing(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_Groupx3_Missing(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # tansfer
        # self.pre_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.post_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # # learnable complementary embeddings
        # self.query_complementary_for_vis = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim,))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.query_complementary_for_ir = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim, ))
        #     for _ in range(num_decoder_layers)
        # ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            # self.query_complementary_for_vis,
            # self.query_complementary_for_ir,
            # self.pre_racq_trans,
            # self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class_visir = self.enc_score_head_visir(output_memory)
        enc_outputs_coord_unact_visir = self.enc_bbox_head_visir(output_memory) + anchors

        if self.training or self.eval_all is True:
            output_memory_vis = paddle.split(output_memory, 2, axis=1)[0]
            output_memory_ir = paddle.split(output_memory, 2, axis=1)[1]
            anchors_vis = paddle.split(anchors, 2, axis=1)[0]
            anchors_ir = paddle.split(anchors, 2, axis=1)[1]

            enc_outputs_class_vis = self.enc_score_head_vis(output_memory_vis)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory_vis) + anchors_vis

            enc_outputs_class_ir = self.enc_score_head_ir(output_memory_ir)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory_ir) + anchors_ir

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1), self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class_visir.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        # topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        # topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact_visir,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class_visir, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_memory_vis, topk_ind_vis)
                target_ir = paddle.gather_nd(output_memory_ir, topk_ind_ir)
                target = paddle.concat([target,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score


@register
class Multi_Groupx3_RTDETRTransformer_Missing(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False):
        super(Multi_Groupx3_RTDETRTransformer_Missing, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx3_Missing_V2(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_Groupx3_Missing_V2(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        # self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        # self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        #
        # self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        # self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # # tansfer
        # self.pre_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.post_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # # learnable complementary embeddings
        # self.query_complementary_for_vis = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim,))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.query_complementary_for_ir = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim, ))
        #     for _ in range(num_decoder_layers)
        # ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        # linear_init_(self.enc_score_head_vis)
        # constant_(self.enc_score_head_vis.bias, bias_cls)
        # constant_(self.enc_bbox_head_vis.layers[-1].weight)
        # constant_(self.enc_bbox_head_vis.layers[-1].bias)
        #
        # linear_init_(self.enc_score_head_ir)
        # constant_(self.enc_score_head_ir.bias, bias_cls)
        # constant_(self.enc_bbox_head_ir.layers[-1].weight)
        # constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i%6](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats)//2:
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats,vis_feats_g,ir_feats_g, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            # visir_feats = vis_feats + ir_feats
            # vis_feats = vis_feats + ir_feats_g
            # ir_feats = vis_feats_g + ir_feats
            all_feats = vis_feats + ir_feats  + vis_feats_g + ir_feats_g
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (all_memory,visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(all_feats)
        visir_spatial_shapes = visir_spatial_shapes[:6]
        visir_level_start_index = visir_level_start_index[:6]
        # (vis_memory, visir_spatial_shapes,
        #  visir_level_start_index) = self._get_encoder_visir_input(vis_feats)
        # (ir_memory, visir_spatial_shapes,
        #  visir_level_start_index) = self._get_encoder_visir_input(ir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            all_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            all_memory,
            # vis_memory,
            # ir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            # self.query_complementary_for_vis,
            # self.query_complementary_for_ir,
            # self.pre_racq_trans,
            # self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, len, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory[:,:len//2,:] = paddle.where(valid_mask, memory[:,:len//2,:], paddle.to_tensor(0.))
        memory[:, len // 2:len, :] = paddle.where(valid_mask, memory[:, len // 2:len, :], paddle.to_tensor(0.))
        # vis_memory = paddle.where(valid_mask, vis_memory, paddle.to_tensor(0.))
        # ir_memory = paddle.where(valid_mask, ir_memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)
        # output_vis_memory = self.enc_output(vis_memory)
        # output_ir_memory = self.enc_output(ir_memory)
        anchors_2 = paddle.concat([anchors,anchors],axis=1)
        enc_outputs_class = self.enc_score_head_visir(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head_visir(output_memory) + anchors_2

        if self.training or self.eval_all is True:
            # enc_outputs_class_vis = self.enc_score_head_visir(output_vis_memory)
            # enc_outputs_coord_unact_vis = self.enc_bbox_head_visir(output_vis_memory) + anchors
            #
            # enc_outputs_class_ir = self.enc_score_head_visir(output_ir_memory)
            # enc_outputs_coord_unact_ir = self.enc_bbox_head_visir(output_ir_memory) + anchors
            enc_outputs_class_vis = paddle.concat([enc_outputs_class[:,:len//4,:],enc_outputs_class[:,-len//4:,:]],axis=1)
            enc_outputs_coord_unact_vis = paddle.concat([enc_outputs_coord_unact[:,:len//4,:],enc_outputs_coord_unact[:,-len//4:,:]],axis=1)
            enc_outputs_class_ir = paddle.concat([enc_outputs_class[:,len//2:3*len//4,:],enc_outputs_class[:,len//4:len//2,:]],axis=1)
            enc_outputs_coord_unact_ir = paddle.concat([enc_outputs_coord_unact[:,len//2:3*len//4,:],enc_outputs_coord_unact[:,len//4:len//2,:]],axis=1)

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1),
                self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class[:,:len//2,:].max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        # topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        # topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact[:,:len//2,:],
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class[:,:len//2,:], topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory[:,:len//2,:], topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(paddle.concat([output_memory[:,:len//4,:],output_memory[:,-len//4:,:]],axis=1), topk_ind_vis)
                target_ir = paddle.gather_nd(paddle.concat([output_memory[:,len//2:3*len//4,:],output_memory[:,len//4:len//2,:]],axis=1),
                                             topk_ind_ir)
                target = paddle.concat([target,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score


@register
class Multi_Groupx3_RTDETRTransformer_Missing_V3(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False):
        super(Multi_Groupx3_RTDETRTransformer_Missing_V3, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        #assert len(backbone_feat_channels) <= num_levels
        backbone_feat_channels = [256,256,256]
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx3_Missing_V3(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_Groupx3_Missing_V3(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # # tansfer
        # self.pre_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.post_racq_trans = nn.LayerList([
        #     copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # # learnable complementary embeddings
        # self.query_complementary_for_vis = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim,))
        #     for _ in range(num_decoder_layers)
        # ])
        #
        # self.query_complementary_for_ir = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, hidden_dim, ))
        #     for _ in range(num_decoder_layers)
        # ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats,vis_feats_g,ir_feats_g, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
            vis_feats = vis_feats + ir_feats_g
            ir_feats = vis_feats_g + ir_feats
            all_feats = vis_feats + ir_feats  + vis_feats_g + ir_feats_g
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        # (all_memory,visir_spatial_shapes,
        #  visir_level_start_index) = self._get_encoder_visir_input(all_feats)
        # visir_spatial_shapes = visir_spatial_shapes[:6]
        # visir_level_start_index = visir_level_start_index[:6]
        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)
        (vis_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(vis_feats)
        (ir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(ir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory,vis_memory,ir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            vis_memory,
            ir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            # self.query_complementary_for_vis,
            # self.query_complementary_for_ir,
            # self.pre_racq_trans,
            # self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta,all_feats)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           vis_memory,
                           ir_memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, len, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        topk_score_r = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        # memory[:,:len//2,:] = paddle.where(valid_mask, memory[:,:len//2,:], paddle.to_tensor(0.))
        # memory[:, len // 2:len, :] = paddle.where(valid_mask, memory[:, len // 2:len, :], paddle.to_tensor(0.))
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        vis_memory = paddle.where(valid_mask, vis_memory, paddle.to_tensor(0.))
        ir_memory = paddle.where(valid_mask, ir_memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)
        output_vis_memory = self.enc_output(vis_memory)
        output_ir_memory = self.enc_output(ir_memory)
        # anchors_2 = paddle.concat([anchors,anchors],axis=1)
        enc_outputs_class = self.enc_score_head_visir(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head_visir(output_memory) + anchors

        if self.training or self.eval_all is True:
            enc_outputs_class_vis = self.enc_score_head_vis(output_vis_memory)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_vis_memory) + anchors

            enc_outputs_class_ir = self.enc_score_head_ir(output_ir_memory)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_ir_memory) + anchors
            # enc_outputs_class_vis = paddle.concat([enc_outputs_class[:,:len//4,:],enc_outputs_class[:,-len//4:,:]],axis=1)
            # enc_outputs_coord_unact_vis = paddle.concat([enc_outputs_coord_unact[:,:len//4,:],enc_outputs_coord_unact[:,-len//4:,:]],axis=1)
            # enc_outputs_class_ir = paddle.concat([enc_outputs_class[:,len//2:3*len//4,:],enc_outputs_class[:,len//4:len//2,:]],axis=1)
            # enc_outputs_coord_unact_ir = paddle.concat([enc_outputs_coord_unact[:,len//2:3*len//4,:],enc_outputs_coord_unact[:,len//4:len//2,:]],axis=1)

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1),
                self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        topk_score_r = topk_score

        # record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent for visir
        # topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        # topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2

        #
        # tensor efficent for vis
        # topk_ind_mask = paddle.where((topk_ind_vis >= visir_level_start_index[0]) & (topk_ind_vis < visir_level_start_index[3]),
        #                              paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        # topk_ind_mask = paddle.where((topk_ind_vis >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2),
        #                              topk_ind_mask)  # 将11~20范围内的位置置为2\

        # topk_score_r = topk_score_vis

        # tensor efficent for ir
        topk_ind_mask = paddle.where(
            (topk_ind_ir >= visir_level_start_index[0]) & (topk_ind_ir < visir_level_start_index[3]),
            paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind_ir >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2),
                                     topk_ind_mask)  # 将11~20范围内的位置置为2
        topk_score_r = topk_score_ir

        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        #
        # #for visir
        # # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        # # topk_score_now = topk_score
        # #
        # # #for vis
        # # real_reference_point = np.array(enc_topk_bboxes_vis[:, :, :2]) * real_hw
        # # topk_score_now = topk_score_vis
        #
        # # #for ir
        # real_reference_point = np.array(enc_topk_bboxes_ir[:, :, :2]) * real_hw
        # topk_score_now = topk_score_ir

        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score_now[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score_now[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/reference_point_ir/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/data/hdd/guojunjie/pp-output-groupx3-missing-v6-vedai640/reference_point_ir/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_vis_memory, topk_ind_vis)
                target_ir = paddle.gather_nd(output_ir_memory,
                                             topk_ind_ir)
                target = paddle.concat([target,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score_r


@register
class Multi_Groupx5_RTDETRTransformer_V3_Missing(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False):
        super(Multi_Groupx5_RTDETRTransformer_V3_Missing, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx5_Missing(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_Groupx5_Missing(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))

        self.enc_score_head_fuse_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_fuse_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # tansfer
        self.pre_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_decoder_layers)
        ])

        self.post_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(num_decoder_layers)
        ])

        # learnable complementary embeddings
        self.query_complementary_for_vis = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries, hidden_dim,))
            for _ in range(num_decoder_layers)
        ])

        self.query_complementary_for_ir = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries, hidden_dim, ))
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_fuse_visir)
        constant_(self.enc_score_head_fuse_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_fuse_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_fuse_visir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, out = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.query_complementary_for_vis,
            self.query_complementary_for_ir,
            self.pre_racq_trans,
            self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits,out, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)
        enc_outputs_class_visir = self.enc_score_head_visir(output_memory)
        enc_outputs_coord_unact_visir = self.enc_bbox_head_visir(output_memory) + anchors

        if self.training or self.eval_all is True:
            output_memory_fuse_visir = paddle.split(output_memory, 2, axis=1)[0] + paddle.split(output_memory, 2, axis=1)[1]
            output_memory_vis = paddle.split(output_memory, 2, axis=1)[0]
            output_memory_ir = paddle.split(output_memory, 2, axis=1)[1]
            anchors_vis = paddle.split(anchors, 2, axis=1)[0]
            anchors_ir = paddle.split(anchors, 2, axis=1)[1]

            enc_outputs_class_fuse_visir = self.enc_score_head_fuse_visir(output_memory_fuse_visir)
            # enc_outputs_coord_unact_fuse_visir = self.enc_bbox_head_fuse_visir(output_memory_fuse_visir) + anchors_vis

            enc_outputs_class_vis = self.enc_score_head_vis(output_memory_vis)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory_vis) + anchors_vis

            enc_outputs_class_ir = self.enc_score_head_ir(output_memory_ir)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory_ir) + anchors_ir

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1), self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class_visir.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        # topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        # topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        #topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact_visir,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)

            reference_points_unact_fuse_vis = reference_points_unact_vis
            reference_points_unact_fuse_ir = reference_points_unact_ir
            enc_topk_bboxes_fuse_vis = enc_topk_bboxes_vis
            enc_topk_bboxes_fuse_ir = enc_topk_bboxes_ir

            # reference_points_unact_fuse_vis = paddle.gather_nd(enc_outputs_coord_unact_fuse_visir,
            #                                               topk_ind_vis)  # unsigmoided.
            # enc_topk_bboxes_fuse_vis = F.sigmoid(reference_points_unact_fuse_vis)
            #
            # reference_points_unact_fuse_ir = paddle.gather_nd(enc_outputs_coord_unact_fuse_visir,
            #                                                    topk_ind_ir)  # unsigmoided.
            # enc_topk_bboxes_fuse_ir = F.sigmoid(reference_points_unact_fuse_ir)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_refrence_point/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_fuse_vis,reference_points_unact_fuse_ir
                    ,reference_points_unact_vis,reference_points_unact_ir], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
            enc_topk_logits_fuse_vis = paddle.gather_nd(enc_outputs_class_fuse_visir, topk_ind_vis)
            enc_topk_logits_fuse_ir = paddle.gather_nd(enc_outputs_class_fuse_visir, topk_ind_ir)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class_visir, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_memory_vis, topk_ind_vis)
                target_ir = paddle.gather_nd(output_memory_ir, topk_ind_ir)
                target_fuse_vis = paddle.gather_nd(output_memory_fuse_visir, topk_ind_vis)
                target_fuse_ir = paddle.gather_nd(output_memory_fuse_visir, topk_ind_ir)
                target = paddle.concat([target,target_fuse_vis,target_fuse_ir,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_fuse_vis,
                                                enc_topk_bboxes_fuse_ir,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_fuse_vis,
                                                 enc_topk_logits_fuse_ir,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score

@register
class Multi_Groupx4_RTDETRTransformer_V3(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False,
                 key_aware = False):
        super(Multi_Groupx4_RTDETRTransformer_V3, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx4(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,key_aware=key_aware)
        self.decoder = TransformerDecoder_Groupx4(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_fvisir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_fvisir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        linear_init_(self.enc_score_head_fvisir)
        constant_(self.enc_score_head_fvisir.bias, bias_cls)
        constant_(self.enc_bbox_head_fvisir.layers[-1].weight)
        constant_(self.enc_bbox_head_fvisir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        output_memory_f = paddle.split(output_memory, 2, axis=1)[0] + paddle.split(output_memory, 2, axis=1)[1]
        output_memory_fvisir = paddle.concat([output_memory_f,output_memory],axis=1)

        anchors_fvisir = paddle.concat([anchors, paddle.split(anchors, 2, axis=1)[0]], axis=1)
        enc_outputs_class_fvisir = self.enc_score_head_fvisir(output_memory_fvisir)
        enc_outputs_coord_unact_fvisir = self.enc_bbox_head_fvisir(output_memory_fvisir) + anchors_fvisir



        if self.training or self.eval_all is True:
            output_memory_vis = paddle.split(output_memory, 2, axis=1)[0]
            output_memory_ir = paddle.split(output_memory, 2, axis=1)[1]
            anchors_vis = paddle.split(anchors, 2, axis=1)[0]
            anchors_ir = paddle.split(anchors, 2, axis=1)[1]

            enc_outputs_class_vis = self.enc_score_head_vis(output_memory_vis)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory_vis) + anchors_vis

            enc_outputs_class_ir = self.enc_score_head_ir(output_memory_ir)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory_ir) + anchors_ir

            enc_outputs_class_f = self.enc_score_head_visir(output_memory_f)
            enc_outputs_coord_unact_f = self.enc_bbox_head_visir(output_memory_f) + anchors_ir

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1), self.num_queries, axis=1)
            topk_score_f, topk_ind_f = paddle.topk(
                enc_outputs_class_f.max(-1), self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class_fvisir.max(-1), self.num_queries, axis=1)

        # ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)

        level1 = visir_level_start_index[1]
        level2 = visir_level_start_index[2] - visir_level_start_index[1]
        level3 = visir_level_start_index[3] - visir_level_start_index[2]
        visir_level_start_index_record = visir_level_start_index + [visir_level_start_index[5]+level3,visir_level_start_index[5]+level3+level1,
                                          visir_level_start_index[5]+level3+level1+level2]
        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index_record[0]) & (topk_ind < visir_level_start_index_record[3]), paddle.full_like(topk_ind_mask, 3), topk_ind_mask)  # 将fusion位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index_record[3]) & (topk_ind < visir_level_start_index_record[6]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将vis位置置为2
        topk_ind_mask = paddle.where(
            (topk_ind >= visir_level_start_index_record[6]),
            paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将ir位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact_fvisir,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            topk_ind_f = paddle.stack([batch_ind, topk_ind_f], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)

            reference_points_unact_f = paddle.gather_nd(enc_outputs_coord_unact_f,
                                                         topk_ind_f)  # unsigmoided.
            enc_topk_bboxes_f = F.sigmoid(reference_points_unact_f)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        # color_p = (128, 0, 128)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 3:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius, color_p, -1)
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius, color_p, -1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_g, -1)
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_g, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/init_reference_point_2/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/init_reference_point_2/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_f,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
            enc_topk_logits_f = paddle.gather_nd(enc_outputs_class_f, topk_ind_f)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class_fvisir, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory_fvisir, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_memory_vis, topk_ind_vis)
                target_ir = paddle.gather_nd(output_memory_ir, topk_ind_ir)
                target_f = paddle.gather_nd(output_memory_f, topk_ind_f)
                target = paddle.concat([target,target_f,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_f,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_f,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score

@register
class Multi_Groupx4_RTDETRTransformer_V3_RANK(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 eval_all = False,
                 key_aware = False):
        super(Multi_Groupx4_RTDETRTransformer_V3_RANK, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.eval_all = eval_all

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Groupx4(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,key_aware=key_aware)
        self.decoder = TransformerDecoder_Groupx4_RANK(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head_vis = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_vis = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_ir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_ir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_visir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_visir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_score_head_fvisir = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head_fvisir = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # query rank layer
        self.rank_aware_content_query = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries * 4, hidden_dim, weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0))))
            for _ in range(num_decoder_layers - 1)
        ])

        self.pre_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_decoder_layers - 1)
        ])

        self.post_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(num_decoder_layers - 1)
        ])

        # Rank-adaptive Classification Head
        self.rank_adaptive_classhead_emb = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries * 4, num_classes, weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0))))
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head_vis)
        constant_(self.enc_score_head_vis.bias, bias_cls)
        constant_(self.enc_bbox_head_vis.layers[-1].weight)
        constant_(self.enc_bbox_head_vis.layers[-1].bias)

        linear_init_(self.enc_score_head_ir)
        constant_(self.enc_score_head_ir.bias, bias_cls)
        constant_(self.enc_bbox_head_ir.layers[-1].weight)
        constant_(self.enc_bbox_head_ir.layers[-1].bias)

        linear_init_(self.enc_score_head_visir)
        constant_(self.enc_score_head_visir.bias, bias_cls)
        constant_(self.enc_bbox_head_visir.layers[-1].weight)
        constant_(self.enc_bbox_head_visir.layers[-1].bias)

        linear_init_(self.enc_score_head_fvisir)
        constant_(self.enc_score_head_fvisir.bias, bias_cls)
        constant_(self.enc_bbox_head_fvisir.layers[-1].weight)
        constant_(self.enc_bbox_head_fvisir.layers[-1].bias)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        for cls_ in self.post_racq_trans:
            linear_init_(cls_)
        for cls_ in self.pre_racq_trans:
            linear_init_(cls_)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.rank_aware_content_query,
            self.rank_adaptive_classhead_emb,
            self.pre_racq_trans,
            self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            eval_all = self.eval_all)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        output_memory_f = paddle.split(output_memory, 2, axis=1)[0] + paddle.split(output_memory, 2, axis=1)[1]
        output_memory_fvisir = paddle.concat([output_memory_f,output_memory],axis=1)

        anchors_fvisir = paddle.concat([anchors, paddle.split(anchors, 2, axis=1)[0]], axis=1)
        enc_outputs_class_fvisir = self.enc_score_head_fvisir(output_memory_fvisir)
        enc_outputs_coord_unact_fvisir = self.enc_bbox_head_fvisir(output_memory_fvisir) + anchors_fvisir



        if self.training or self.eval_all is True:
            output_memory_vis = paddle.split(output_memory, 2, axis=1)[0]
            output_memory_ir = paddle.split(output_memory, 2, axis=1)[1]
            anchors_vis = paddle.split(anchors, 2, axis=1)[0]
            anchors_ir = paddle.split(anchors, 2, axis=1)[1]

            enc_outputs_class_vis = self.enc_score_head_vis(output_memory_vis)
            enc_outputs_coord_unact_vis = self.enc_bbox_head_vis(output_memory_vis) + anchors_vis

            enc_outputs_class_ir = self.enc_score_head_ir(output_memory_ir)
            enc_outputs_coord_unact_ir = self.enc_bbox_head_ir(output_memory_ir) + anchors_ir

            enc_outputs_class_f = self.enc_score_head_visir(output_memory_f)
            enc_outputs_coord_unact_f = self.enc_bbox_head_visir(output_memory_f) + anchors_ir

            topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class_vis.max(-1), self.num_queries, axis=1)
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class_ir.max(-1), self.num_queries, axis=1)
            topk_score_f, topk_ind_f = paddle.topk(
                enc_outputs_class_f.max(-1), self.num_queries, axis=1)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class_fvisir.max(-1), self.num_queries, axis=1)

        # ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)

        level1 = visir_level_start_index[1]
        level2 = visir_level_start_index[2] - visir_level_start_index[1]
        level3 = visir_level_start_index[3] - visir_level_start_index[2]
        visir_level_start_index_record = visir_level_start_index + [visir_level_start_index[5]+level3,visir_level_start_index[5]+level3+level1,
                                          visir_level_start_index[5]+level3+level1+level2]
        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index_record[0]) & (topk_ind < visir_level_start_index_record[3]), paddle.full_like(topk_ind_mask, 3), topk_ind_mask)  # 将fusion位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index_record[3]) & (topk_ind < visir_level_start_index_record[6]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将vis位置置为2
        topk_ind_mask = paddle.where(
            (topk_ind >= visir_level_start_index_record[6]),
            paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将ir位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact_fvisir,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if self.training or self.eval_all is True:
            topk_ind_vis = paddle.stack([batch_ind, topk_ind_vis], axis=-1)
            topk_ind_ir = paddle.stack([batch_ind, topk_ind_ir], axis=-1)
            topk_ind_f = paddle.stack([batch_ind, topk_ind_f], axis=-1)
            reference_points_unact_vis = paddle.gather_nd(enc_outputs_coord_unact_vis,
                                                      topk_ind_vis)  # unsigmoided.
            enc_topk_bboxes_vis = F.sigmoid(reference_points_unact_vis)

            reference_points_unact_ir = paddle.gather_nd(enc_outputs_coord_unact_ir,
                                                          topk_ind_ir)  # unsigmoided.
            enc_topk_bboxes_ir = F.sigmoid(reference_points_unact_ir)

            reference_points_unact_f = paddle.gather_nd(enc_outputs_coord_unact_f,
                                                         topk_ind_f)  # unsigmoided.
            enc_topk_bboxes_f = F.sigmoid(reference_points_unact_f)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        # color_p = (128, 0, 128)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 3:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius, color_p, -1)
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius, color_p, -1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_g, -1)
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_g, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/init_reference_point_2/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-groupx4-detrv3-newm3fd/init_reference_point_2/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training or self.eval_all is True:
            reference_points_unact = paddle.concat(
                [reference_points_unact,reference_points_unact_f,reference_points_unact_vis,reference_points_unact_ir ], 1)
            if self.training:
                reference_points_unact = reference_points_unact.detach()

            enc_topk_logits_vis = paddle.gather_nd(enc_outputs_class_vis, topk_ind_vis)
            enc_topk_logits_ir = paddle.gather_nd(enc_outputs_class_ir, topk_ind_ir)
            enc_topk_logits_f = paddle.gather_nd(enc_outputs_class_f, topk_ind_f)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class_fvisir, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory_fvisir, topk_ind)
            if self.training or self.eval_all is True:
                target_vis = paddle.gather_nd(output_memory_vis, topk_ind_vis)
                target_ir = paddle.gather_nd(output_memory_ir, topk_ind_ir)
                target_f = paddle.gather_nd(output_memory_f, topk_ind_f)
                target = paddle.concat([target,target_f,target_vis,target_ir],1)
                if self.training:
                    target = target.detach()

                enc_topk_bboxes = paddle.concat([enc_topk_bboxes,enc_topk_bboxes_f,enc_topk_bboxes_vis,enc_topk_bboxes_ir],1)
                enc_topk_logits = paddle.concat([enc_topk_logits,enc_topk_logits_f,enc_topk_logits_vis,enc_topk_logits_ir],1)

        if denoising_class is not None:
            #[DN,visir,vis,ir]
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score

@register
class Multi_RTDETRTransformer_V3(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware = False,
                 proj_all = False,
                 anchor_grid_size=0.05):
        super(Multi_RTDETRTransformer_V3, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all
        self.anchor_grid_size = anchor_grid_size

        if self.proj_all:
            # backbone feature projection
            self._build_input_proj_layer(backbone_visir_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,key_aware=key_aware)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        if self.proj_all:
            for l in self.input_proj:
                xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]

        if self.proj_all:
            (memory, spatial_shapes,
             level_start_index) = self._get_encoder_input(visir_feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        if self.proj_all:
            target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
                self._get_decoder_input(gt_meta,
                memory, spatial_shapes, level_start_index, denoising_class, denoising_bbox_unact)
        else:
            target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score = \
                self._get_decoder_input(gt_meta,
                                        visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class,
                                        denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize with box
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw2 = [w,h,w,h]
        # real_hw = np.array(real_hw)
        # real_hw2 = np.array(real_hw2)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        # real_bbox = np.array(enc_topk_bboxes) * real_hw2
        #
        #   ##plot point and box
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+3,color_r,-1)
        #
        #                 # x = round(real_bbox[xx][[ii][0]-real_bbox[xx][ii][2]/2])
        #                 # x = round(real_bbox[xx][[ii][1]+real_bbox[xx][ii][3]/2])
        #                 # x = round(real_bbox[xx][[ii][0]+real_bbox[xx][ii][2]/2])
        #                 # x= round(real_bbox[xx][[ii][1]-real_bbox[xx][ii][3]/2])
        #
        #                 cv2.rectangle(vis_imgs[xx],(round(real_bbox[xx][ii][0]-real_bbox[xx][ii][2]/2),round(real_bbox[xx][ii][1]+real_bbox[xx][ii][3]/2)),
        #                               (round(real_bbox[xx][ii][0]+real_bbox[xx][ii][2]/2),round(real_bbox[xx][ii][1]-real_bbox[xx][ii][3]/2)),color_r,2)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+3,color_r,-1)
        #
        #                 cv2.rectangle(ir_imgs[xx], (round(real_bbox[xx][ii][0] - real_bbox[xx][ii][2] / 2),
        #                                              round(real_bbox[xx][ii][1] + real_bbox[xx][ii][3] / 2)),
        #                               (round(real_bbox[xx][ii][0] + real_bbox[xx][ii][2] / 2),
        #                                round(real_bbox[xx][ii][1] - real_bbox[xx][ii][3] / 2)), color_r, 2)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_reference_point_box/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_reference_point_box/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score
class TransformerDecoderLayer_Rotate(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 angle_max=None,
                 angle_proj=None,
                 weight_attr=None,
                 bias_attr=None,
                 key_aware=False,
                 split_attention=False):
        super(TransformerDecoderLayer_Rotate, self).__init__()

        self.angle_max = angle_max
        self.angle_proj = angle_proj
        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention

        self.cross_attn = PPMSDeformableAttention_Rotate(d_model, n_head, n_levels,
                                                  n_points, 1.0)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                gt_meta,
                tgt,
                reference_points,
                angle,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                angle_max,
                half_pi_bin,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                topk_ind_mask = None,
                topk_score = None,
                mask_vis = None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(gt_meta,
            self.with_pos_embed(tgt, query_pos_embed), reference_points, angle, memory,
            memory_spatial_shapes, memory_level_start_index,angle_max,half_pi_bin, memory_mask,topk_ind_mask,topk_score, mask_vis=mask_vis)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt
@register
class Multi_RTDETRTransformer_RadarCamera(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_radarcamera_feat_channels=[256, 256, 256, 256, 256, 256],
                 radarcamera_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_radarcamera_levels=6,
                 anchor_grid_size=0.05,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware = False,
                 proj_all = False):
        super(Multi_RTDETRTransformer_RadarCamera, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.radarcamera_feat_strides = radarcamera_feat_strides
        self.num_levels = num_levels
        self.num_radarcamera_levels = num_radarcamera_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all
        self.anchor_grid_size = anchor_grid_size
        if self.proj_all:
            # backbone feature projection
            self._build_input_proj_layer(backbone_radarcamera_feat_channels)

        self._build_radarcamera_input_proj_layer(backbone_radarcamera_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_RadarCamera(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_radarcamera_levels,
            num_decoder_points,key_aware=key_aware)
        self.decoder = TransformerDecoder_RadarCamera(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_radarcamera:
            xavier_uniform_(l[0].weight)

        if self.proj_all:
            for l in self.input_proj:
                xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_radarcamera_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_radarcamera = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_radarcamera.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_radarcamera_levels - len(backbone_feat_channels)):
            self.input_proj_radarcamera.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_radarcamera_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_radarcamera[i](feat) for i, feat in enumerate(feats)]
        if self.num_radarcamera_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_radarcamera_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_radarcamera[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_radarcamera[i](proj_feats[-1]))

        proj_feats_radar = proj_feats[:3]
        proj_feats_camera = proj_feats[3:]

        # get encoder inputs radar
        feat_flatten_radar = []
        spatial_shapes_radar = []
        level_start_index_radar = [0, ]
        for i, feat in enumerate(proj_feats_radar):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten_radar.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes_radar.append([h, w])
            # [l], start index of each level
            level_start_index_radar.append(h * w + level_start_index_radar[-1])

        # [b, l, c]
        feat_flatten_radar = paddle.concat(feat_flatten_radar, 1)
        level_start_index_radar.pop()

        # get encoder inputs camera
        feat_flatten_camera = []
        spatial_shapes_camera = []
        level_start_index_camera = [0, ]
        for i, feat in enumerate(proj_feats_camera):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten_camera.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes_camera.append([h, w])
            # [l], start index of each level
            level_start_index_camera.append(h * w + level_start_index_camera[-1])

        # [b, l, c]
        feat_flatten_camera = paddle.concat(feat_flatten_camera, 1)
        level_start_index_camera.pop()

        # print('DEBUG: spatial_shapes_radar in _get_encoder_radarcamera_input:', spatial_shapes_radar)
        # print('DEBUG: spatial_shapes_camera in _get_encoder_radarcamera_input:', spatial_shapes_camera)
        # print('DEBUG: feat_flatten_radar.shape:', feat_flatten_radar.shape)
        # print('DEBUG: feat_flatten_camera.shape:', feat_flatten_camera.shape)
        return (feat_flatten_radar, spatial_shapes_radar, level_start_index_radar,feat_flatten_camera, spatial_shapes_camera, level_start_index_camera)



    def forward(self, feats, radar_feats, camera_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding

        radarcamera_feats = radar_feats + camera_feats

        (radar_memory, radar_spatial_shapes,
         radar_level_start_index,camera_memory, camera_spatial_shapes,
         camera_level_start_index) = self._get_encoder_radarcamera_input(radarcamera_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None


        target_radar, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
                                    radar_memory, camera_memory, radar_spatial_shapes, radar_level_start_index, denoising_class,
                                    denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target_radar,
            init_ref_points_unact,
            radar_memory,
            camera_memory,
            radar_spatial_shapes,
            radar_level_start_index,
            camera_spatial_shapes,
            camera_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.radarcamera_feat_strides[:3]
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * self.anchor_grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           radar_memory,
                           camera_memory,
                           radar_spatial_shapes,
                           radar_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        radar_bs, _, _ = radar_memory.shape
        camera_bs, _, _ = camera_memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            #print('DEBUG: radar_spatial_shapes in _get_decoder_input:', radar_spatial_shapes)
            anchors, valid_mask = self._generate_anchors(radar_spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        # 确保锚点维度与雷达内存维度匹配
        if radar_memory.shape[1] != valid_mask.shape[1]:
            # print('DEBUG shape mismatch:', radar_memory.shape, valid_mask.shape,
            #       radar_spatial_shapes)
            # 截取valid_mask和anchors以匹配radar_memory的维度
            valid_mask = valid_mask[:, :radar_memory.shape[1], :]
            anchors = anchors[:, :radar_memory.shape[1], :]
        radar_memory = paddle.where(valid_mask, radar_memory, paddle.to_tensor(0.))
        output_radar_memory = self.enc_output(radar_memory)

        #output_camera_memory = self.enc_output(camera_memory)

        enc_outputs_class = self.enc_score_head(output_radar_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_radar_memory) + anchors

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((radar_bs,300)))
        topk_ind_mask.stop_gradient = True

        # extract region proposal boxes
        batch_ind = paddle.arange(end=radar_bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            radar_target = self.tgt_embed.weight.unsqueeze(0).tile([radar_bs, 1, 1])
        else:
            radar_target = paddle.gather_nd(output_radar_memory, topk_ind)
            if self.training:
                radar_target = radar_target.detach()
                #camera_target = camera_target.detach()
        if denoising_class is not None:
            radar_target = paddle.concat([denoising_class, radar_target], 1)

        return radar_target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score


@register
class Multi_RTDETRTransformer_RadarCamera_Rotate(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 backbone_radarcamera_feat_channels=[256, 256, 256, 256, 256, 256],
                 radarcamera_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_radarcamera_levels=6,
                 anchor_grid_size=0.05,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 angle_noise_scale=0.03,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware=False,
                 proj_all=False):
        super(Multi_RTDETRTransformer_RadarCamera_Rotate, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.radarcamera_feat_strides = radarcamera_feat_strides
        self.num_levels = num_levels
        self.num_radarcamera_levels = num_radarcamera_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all
        self.anchor_grid_size = anchor_grid_size

        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.angle_max = 90
        self.half_pi_bin = self.half_pi / self.angle_max
        angle_proj = paddle.linspace(
            0, self.angle_max, self.angle_max + 1, dtype='float32')
        self.angle_proj = angle_proj * self.half_pi_bin

        if self.proj_all:
            self._build_input_proj_layer(backbone_radarcamera_feat_channels)
        self._build_radarcamera_input_proj_layer(
            backbone_radarcamera_feat_channels)

        decoder_layer = TransformerDecoderLayer_RadarCamera_Rotate(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_radarcamera_levels, num_decoder_points, self.angle_max,
            self.angle_proj, key_aware=key_aware)
        self.decoder = TransformerDecoder_RadarCamera_Rotate(
            hidden_dim, decoder_layer, num_decoder_layers, self.angle_max,
            self.angle_proj, self.half_pi_bin, eval_idx)

        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.angle_noise_scale = angle_noise_scale

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.enc_angle_head = nn.Linear(hidden_dim, self.angle_max + 1)

        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_angle_head = nn.LayerList([
            nn.Linear(hidden_dim, self.angle_max + 1)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_angle = [10.] + [1.] * self.angle_max
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        constant_(self.enc_angle_head.weight)
        vector_(self.enc_angle_head.bias, bias_angle)
        for cls_, reg_, angle_ in zip(self.dec_score_head, self.dec_bbox_head,
                                      self.dec_angle_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)
            constant_(angle_.weight)
            vector_(angle_.bias, bias_angle)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj_radarcamera:
            xavier_uniform_(layer[0].weight)
        if self.proj_all:
            for layer in self.input_proj:
                xavier_uniform_(layer[0].weight)

        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv',
                     nn.Conv2D(
                         in_channels,
                         self.hidden_dim,
                         kernel_size=1,
                         bias_attr=False)),
                    ('norm',
                     nn.BatchNorm2D(
                         self.hidden_dim,
                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv',
                     nn.Conv2D(
                         in_channels,
                         self.hidden_dim,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         bias_attr=False)),
                    ('norm',
                     nn.BatchNorm2D(
                         self.hidden_dim,
                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_radarcamera_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_radarcamera = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_radarcamera.append(
                nn.Sequential(
                    ('conv',
                     nn.Conv2D(
                         in_channels,
                         self.hidden_dim,
                         kernel_size=1,
                         bias_attr=False)),
                    ('norm',
                     nn.BatchNorm2D(
                         self.hidden_dim,
                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_radarcamera_levels - len(backbone_feat_channels)):
            self.input_proj_radarcamera.append(
                nn.Sequential(
                    ('conv',
                     nn.Conv2D(
                         in_channels,
                         self.hidden_dim,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         bias_attr=False)),
                    ('norm',
                     nn.BatchNorm2D(
                         self.hidden_dim,
                         weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_radarcamera_input(self, feats):
        proj_feats = [
            self.input_proj_radarcamera[i](feat)
            for i, feat in enumerate(feats)
        ]
        if self.num_radarcamera_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_radarcamera_levels):
                if i == len_srcs:
                    proj_feats.append(
                        self.input_proj_radarcamera[i](feats[-1]))
                else:
                    proj_feats.append(
                        self.input_proj_radarcamera[i](proj_feats[-1]))

        proj_feats_radar = proj_feats[:3]
        proj_feats_camera = proj_feats[3:]

        feat_flatten_radar = []
        spatial_shapes_radar = []
        level_start_index_radar = [0, ]
        for feat in proj_feats_radar:
            _, _, h, w = feat.shape
            feat_flatten_radar.append(feat.flatten(2).transpose([0, 2, 1]))
            spatial_shapes_radar.append([h, w])
            level_start_index_radar.append(h * w + level_start_index_radar[-1])
        feat_flatten_radar = paddle.concat(feat_flatten_radar, 1)
        level_start_index_radar.pop()

        feat_flatten_camera = []
        spatial_shapes_camera = []
        level_start_index_camera = [0, ]
        for feat in proj_feats_camera:
            _, _, h, w = feat.shape
            feat_flatten_camera.append(feat.flatten(2).transpose([0, 2, 1]))
            spatial_shapes_camera.append([h, w])
            level_start_index_camera.append(h * w + level_start_index_camera[-1])
        feat_flatten_camera = paddle.concat(feat_flatten_camera, 1)
        level_start_index_camera.pop()

        return (feat_flatten_radar, spatial_shapes_radar, level_start_index_radar,
                feat_flatten_camera, spatial_shapes_camera,
                level_start_index_camera)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          dtype='float32'):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.radarcamera_feat_strides[:3]
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(end=h, dtype=dtype),
                paddle.arange(end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)
            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * self.anchor_grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))
        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float('inf')))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           radar_memory,
                           camera_memory,
                           radar_spatial_shapes,
                           radar_level_start_index,
                           camera_spatial_shapes,
                           camera_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           denosing_angle=None):
        radar_bs, _, _ = radar_memory.shape
        topk_score = None
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(radar_spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if radar_memory.shape[1] != valid_mask.shape[1]:
            valid_mask = valid_mask[:, :radar_memory.shape[1], :]
            anchors = anchors[:, :radar_memory.shape[1], :]
        radar_memory = paddle.where(valid_mask, radar_memory,
                                    paddle.to_tensor(0.))
        output_radar_memory = self.enc_output(radar_memory)

        enc_outputs_class = self.enc_score_head(output_radar_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_radar_memory) + anchors
        enc_outputs_angle_cls = self.enc_angle_head(output_radar_memory)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        topk_ind_mask = paddle.to_tensor(
            np.zeros((radar_bs, self.num_queries)))
        topk_ind_mask.stop_gradient = True

        batch_ind = paddle.arange(end=radar_bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        enc_topk_angles_cls = paddle.gather_nd(enc_outputs_angle_cls, topk_ind)
        reference_angle = F.softmax(
            enc_topk_angles_cls.reshape(
                [radar_bs, self.num_queries, 1, self.angle_max + 1]),
            axis=-1).matmul(self.angle_proj)

        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            if denosing_angle is None:
                denosing_angle = paddle.zeros(
                    [radar_bs, denoising_bbox_unact.shape[1], 1],
                    dtype=reference_angle.dtype)
            reference_angle = paddle.concat(
                [denosing_angle, reference_angle], 1)

        if self.training:
            reference_points_unact = reference_points_unact.detach()
            reference_angle_cls = enc_topk_angles_cls.detach()
            reference_angle = reference_angle.detach()
        else:
            reference_angle_cls = enc_topk_angles_cls
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        if self.learnt_init_query:
            radar_target = self.tgt_embed.weight.unsqueeze(0).tile(
                [radar_bs, 1, 1])
        else:
            radar_target = paddle.gather_nd(output_radar_memory, topk_ind)
            if self.training:
                radar_target = radar_target.detach()
        if denoising_class is not None:
            radar_target = paddle.concat([denoising_class, radar_target], 1)

        return (
            radar_target,
            reference_points_unact,
            reference_angle_cls,
            reference_angle,
            enc_topk_bboxes,
            enc_topk_logits,
            enc_topk_angles_cls,
            topk_ind_mask,
            topk_score,
            None,
        )

    def forward(self,
                feats,
                radar_feats,
                camera_feats,
                pad_mask=None,
                gt_meta=None,
                topk_ind_mask=None,
                topk_score=None):
        radarcamera_feats = radar_feats + camera_feats

        (radar_memory, radar_spatial_shapes, radar_level_start_index,
         camera_memory, camera_spatial_shapes,
         camera_level_start_index) = self._get_encoder_radarcamera_input(
            radarcamera_feats)

        if self.training:
            try:
                denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group_rotated(
                        gt_meta,
                        self.num_classes,
                        self.num_queries,
                        self.denoising_class_embed.weight,
                        self.num_denoising,
                        self.label_noise_ratio,
                        self.box_noise_scale,
                        self.angle_noise_scale)
            except NotImplementedError:
                denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(
                        gt_meta,
                        self.num_classes,
                        self.num_queries,
                        self.denoising_class_embed.weight,
                        self.num_denoising,
                        self.label_noise_ratio,
                        self.box_noise_scale)
                denosing_angle = None
        else:
            denoising_class = denoising_bbox_unact = denosing_angle = None
            attn_mask = dn_meta = None

        (target_radar, init_ref_points_unact, init_ref_angle_cls,
         init_ref_angle, enc_topk_bboxes, enc_topk_logits,
         enc_topk_angles_cls, topk_ind_mask, topk_score,
         mask_vis) = self._get_decoder_input(
            gt_meta,
            radar_memory,
            camera_memory,
            radar_spatial_shapes,
            radar_level_start_index,
            camera_spatial_shapes,
            camera_level_start_index,
            denoising_class,
            denoising_bbox_unact,
            denosing_angle)

        out_bboxes, out_logits, out_angles_cls, out_angles = self.decoder(
            gt_meta,
            target_radar,
            init_ref_points_unact,
            init_ref_angle_cls,
            init_ref_angle,
            radar_memory,
            camera_memory,
            radar_spatial_shapes,
            radar_level_start_index,
            camera_spatial_shapes,
            camera_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_angle_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask=topk_ind_mask,
            topk_score=topk_score,
            mask_vis=mask_vis)

        return (out_bboxes, out_logits, out_angles_cls, out_angles,
                enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls,
                self.angle_max, self.angle_proj, dn_meta)


@register
class RTDETRTransformer_Rotate(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 angle_noise_scale=0.03,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(RTDETRTransformer_Rotate, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.angle_max = 90
        self.half_pi_bin = self.half_pi / self.angle_max
        angle_proj = paddle.linspace(0, self.angle_max, self.angle_max + 1)
        self.angle_proj = angle_proj * self.half_pi_bin

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Rotate(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points,self.angle_max, self.angle_proj)
        self.decoder = TransformerDecoder_Rotate(hidden_dim, decoder_layer,
                                          num_decoder_layers,self.angle_max, self.angle_proj ,self.half_pi_bin ,eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.angle_noise_scale = angle_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        # angle head
        self.enc_angle_head = nn.Linear(hidden_dim, self.angle_max + 1)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        #angle head
        self.dec_angle_head = nn.LayerList([
            nn.Linear(hidden_dim, self.angle_max + 1)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        bias_angle = [10.] + [1.] * self.angle_max
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_angle_head.weight)
        vector_(self.enc_angle_head.bias, bias_angle)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_, angle_ in zip(self.dec_score_head, self.dec_bbox_head, self.dec_angle_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)
            constant_(angle_.weight)
            vector_(angle_.bias, bias_angle)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def forward(self, feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_rotated(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale,
                                            self.angle_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = None, None, None, None, None

        target, init_ref_points_unact,init_ref_angle_cls, init_ref_angle, enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls, topk_score = \
            self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact, denosing_angle)

        # decoder
        out_bboxes, out_logits, out_angles_cls, out_angles = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            init_ref_angle_cls,
            init_ref_angle,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_angle_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask=topk_ind_mask,
            topk_score=topk_score)

        return (out_bboxes, out_logits, out_angles_cls, out_angles, enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls,self.angle_max,self.angle_proj,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           denosing_angle=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        #pred angle
        enc_outputs_angle_cls = self.enc_angle_head(output_memory)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        # angle
        enc_topk_angles = paddle.gather_nd(enc_outputs_angle_cls, topk_ind)

        # get angle
        b, l = enc_topk_angles.shape[:2]
        reference_angle = F.softmax(enc_topk_angles.reshape([b, l, 1, self.angle_max + 1
                                                 ])).matmul(self.angle_proj)

        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            reference_angle = paddle.concat(
                [denosing_angle, reference_angle], 1
            )

        if self.training:
            reference_points_unact = reference_points_unact.detach()
            reference_angle_cls = enc_topk_angles.detach()
            reference_angle = reference_angle.detach()
        else:
            reference_angle_cls = enc_topk_angles
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, reference_angle_cls, reference_angle, enc_topk_bboxes, enc_topk_logits, enc_topk_angles, topk_score


@register
class RTDETRTransformer_Rotate_RouteROI(RTDETRTransformer_Rotate):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[256, 256, 256],
                 feat_strides=[4, 8, 16],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 angle_noise_scale=0.03,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 proposal_topk_per_level=[180, 90, 30],
                 proposal_level_score_bias=[0.25, 0.10, 0.0],
                 proposal_small_box_threshold=0.025,
                 proposal_small_box_bonus_weight=0.12,
                 proposal_use_learned_rerank=True,
                 proposal_rank_score_weight=1.0):
        super(RTDETRTransformer_Rotate_RouteROI, self).__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            position_embed_type=position_embed_type,
            backbone_feat_channels=backbone_feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_decoder_points=num_decoder_points,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            angle_noise_scale=angle_noise_scale,
            learnt_init_query=learnt_init_query,
            eval_size=eval_size,
            eval_idx=eval_idx,
            eps=eps)
        self.proposal_topk_per_level = proposal_topk_per_level
        self.proposal_level_score_bias = proposal_level_score_bias
        self.proposal_small_box_threshold = proposal_small_box_threshold
        self.proposal_small_box_bonus_weight = proposal_small_box_bonus_weight
        self.proposal_use_learned_rerank = proposal_use_learned_rerank
        self.proposal_rank_score_weight = proposal_rank_score_weight
        self.enc_rank_head = nn.Linear(hidden_dim, 1)
        linear_init_(self.enc_rank_head)

    def _select_topk_queries(self,
                             output_memory,
                             enc_outputs_class,
                             enc_outputs_coord_unact,
                             spatial_shapes,
                             level_start_index):
        bs = output_memory.shape[0]
        cls_score = enc_outputs_class.max(-1)
        boxes = F.sigmoid(enc_outputs_coord_unact)
        if self.proposal_use_learned_rerank:
            rank_score = self.enc_rank_head(output_memory).squeeze(-1)
        else:
            rank_score = paddle.zeros_like(cls_score)

        candidate_scores = []
        candidate_indices = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            if lvl >= len(self.proposal_topk_per_level):
                break
            start = level_start_index[lvl]
            end = start + h * w
            level_score = cls_score[:, start:end]
            if lvl < len(self.proposal_level_score_bias):
                level_score = level_score + self.proposal_level_score_bias[lvl]
            small_mask = paddle.minimum(boxes[:, start:end, 2],
                                        boxes[:, start:end, 3]) < self.proposal_small_box_threshold
            level_score = level_score + small_mask.astype(level_score.dtype) * \
                self.proposal_small_box_bonus_weight
            if self.proposal_use_learned_rerank:
                level_score = level_score + self.proposal_rank_score_weight * rank_score[:, start:end]
            topk_per_level = min(self.proposal_topk_per_level[lvl], level_score.shape[1])
            level_topk_score, level_topk_idx = paddle.topk(
                level_score, topk_per_level, axis=1)
            candidate_scores.append(level_topk_score)
            candidate_indices.append(level_topk_idx + start)

        candidate_scores = paddle.concat(candidate_scores, axis=1)
        candidate_indices = paddle.concat(candidate_indices, axis=1)
        final_topk = min(self.num_queries, candidate_scores.shape[1])
        topk_score, topk_order = paddle.topk(candidate_scores, final_topk, axis=1)
        batch_idx = paddle.arange(end=bs, dtype=topk_order.dtype).unsqueeze(-1).tile(
            [1, final_topk])
        gather_idx = paddle.stack([batch_idx, topk_order], axis=-1)
        topk_indices = paddle.gather_nd(candidate_indices, gather_idx)
        return topk_score, topk_indices

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           denosing_angle=None):
        bs, _, _ = memory.shape
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        enc_outputs_angle_cls = self.enc_angle_head(output_memory)

        level_start_index = [0]
        for h, w in spatial_shapes[:-1]:
            level_start_index.append(level_start_index[-1] + h * w)
        topk_score, topk_index = self._select_topk_queries(
            output_memory, enc_outputs_class, enc_outputs_coord_unact,
            spatial_shapes, level_start_index)

        batch_ind = paddle.arange(end=bs, dtype=topk_index.dtype).unsqueeze(-1).tile(
            [1, topk_index.shape[1]])
        topk_ind = paddle.stack([batch_ind, topk_index], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact, topk_ind)
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_angles = paddle.gather_nd(enc_outputs_angle_cls, topk_ind)

        b, l = enc_topk_angles.shape[:2]
        reference_angle = F.softmax(
            enc_topk_angles.reshape([b, l, 1, self.angle_max + 1]),
            axis=-1).matmul(self.angle_proj)

        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            reference_angle = paddle.concat([denosing_angle, reference_angle], 1)

        if self.training:
            reference_points_unact = reference_points_unact.detach()
            reference_angle_cls = enc_topk_angles.detach()
            reference_angle = reference_angle.detach()
        else:
            reference_angle_cls = enc_topk_angles
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return (target, reference_points_unact, reference_angle_cls,
                reference_angle, enc_topk_bboxes, enc_topk_logits,
                enc_topk_angles, topk_score)

    def forward(self, feats, pad_mask=None, gt_meta=None, topk_ind_mask=None, topk_score=None):
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(feats)

        if self.training:
            denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_rotated(
                    gt_meta,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed.weight,
                    self.num_denoising,
                    self.label_noise_ratio,
                    self.box_noise_scale,
                    self.angle_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = \
                None, None, None, None, None

        target, init_ref_points_unact, init_ref_angle_cls, init_ref_angle, \
            enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls, topk_score = \
            self._get_decoder_input(
                memory, spatial_shapes, denoising_class, denoising_bbox_unact,
                denosing_angle)

        out_bboxes, out_logits, out_angles_cls, out_angles, last_query_feat = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            init_ref_angle_cls,
            init_ref_angle,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_angle_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask=topk_ind_mask,
            topk_score=topk_score,
            return_last_query=True)

        return (out_bboxes, out_logits, out_angles_cls, out_angles,
                enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls,
                self.angle_max, self.angle_proj, dn_meta, last_query_feat)


@register
class Multi_RTDETRTransformer_Rotate(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 angle_noise_scale=0.03,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware = False,
                 proj_all = False,
                 split_attention = False):
        super(Multi_RTDETRTransformer_Rotate, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all

        self.half_pi = paddle.to_tensor(
            [1.5707963267948966], dtype=paddle.float32)
        self.angle_max = 90
        self.half_pi_bin = self.half_pi / self.angle_max
        angle_proj = paddle.linspace(0, self.angle_max, self.angle_max + 1)
        self.angle_proj = angle_proj * self.half_pi_bin


        if self.proj_all:
            # backbone feature projection
            self._build_input_proj_layer(backbone_visir_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_Rotate(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,self.angle_max, self.angle_proj,key_aware=key_aware,split_attention=split_attention)
        self.decoder = TransformerDecoder_Rotate(hidden_dim, decoder_layer,
                                          num_decoder_layers,self.angle_max, self.angle_proj ,self.half_pi_bin, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.angle_noise_scale = angle_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # angle head
        self.enc_angle_head = nn.Linear(hidden_dim, self.angle_max + 1)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # angle head
        self.dec_angle_head = nn.LayerList([
            nn.Linear(hidden_dim, self.angle_max + 1)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        bias_angle = [10.] + [1.] * self.angle_max
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_angle_head.weight)
        vector_(self.enc_angle_head.bias, bias_angle)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_, angle_ in zip(self.dec_score_head, self.dec_bbox_head, self.dec_angle_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)
            constant_(angle_.weight)
            vector_(angle_.bias, bias_angle)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        # if self.num_visir_levels == 6:
        visir_feats = vis_feats + ir_feats
        # elif self.num_visir_levels == 5:
        #     visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]

        if self.proj_all:
            (memory, spatial_shapes,
             level_start_index) = self._get_encoder_input(visir_feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact,denosing_angle, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_rotated(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale,
                                            self.angle_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, denosing_angle, attn_mask, dn_meta = None, None, None, None, None

        if self.proj_all:
            target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score, mask_vis = \
                self._get_decoder_input(gt_meta,
                memory, spatial_shapes, level_start_index, denoising_class, denoising_bbox_unact)
        else:
            target, init_ref_points_unact,init_ref_angle_cls, init_ref_angle, enc_topk_bboxes, enc_topk_logits,enc_topk_angles_cls, topk_ind_mask, topk_score, mask_vis = \
                self._get_decoder_input(gt_meta,
                                        visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class,
                                        denoising_bbox_unact,denosing_angle)

        # decoder
        out_bboxes, out_logits,out_angles_cls, out_angles = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            init_ref_angle_cls,
            init_ref_angle,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_angle_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score,
            mask_vis = mask_vis)
        return (out_bboxes, out_logits, out_angles_cls, out_angles, enc_topk_bboxes, enc_topk_logits, enc_topk_angles_cls,self.angle_max,self.angle_proj,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           denosing_angle=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        # pred angle
        enc_outputs_angle_cls = self.enc_angle_head(output_memory)

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True

        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        # select modality reference position to caculate loss
        mask_vis = topk_ind_mask == 1
        mask_vis = paddle.unsqueeze(mask_vis, axis=-1)
        mask_vis = paddle.broadcast_to(mask_vis, shape=[bs, 300, 4])


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        # angle
        enc_topk_angles = paddle.gather_nd(enc_outputs_angle_cls, topk_ind)

        # get angle
        b, l = enc_topk_angles.shape[:2]
        reference_angle = F.softmax(enc_topk_angles.reshape([b, l, 1, self.angle_max + 1
                                                             ])).matmul(self.angle_proj)

        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            reference_angle = paddle.concat(
                [denosing_angle, reference_angle], 1
            )
        if self.training:
            reference_points_unact = reference_points_unact.detach()
            reference_angle_cls = enc_topk_angles.detach()
            reference_angle = reference_angle.detach()
        else:
            reference_angle_cls = enc_topk_angles
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact,reference_angle_cls, reference_angle, enc_topk_bboxes, enc_topk_logits, enc_topk_angles, topk_ind_mask, topk_score, mask_vis


@register
class Multi_RTDETRTransformer_V7(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 backbone_query_feat_channels=[256, 256, 256, 256, 256, 256, 256, 256, 256],
                 query_feat_strides=[8, 16, 32, 8, 16, 32, 8, 16, 32],
                 num_query_levels=9,

                 average_query = False,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer_V7, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)


        self.average_query = average_query
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.query_feat_strides = query_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_query_levels = num_query_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size

        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_query_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_query_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_query_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
            query_feats = visir_feats + [vis_feats[i] + ir_feats[i] for i in range(3)]
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (query_memory, query_spatial_shapes,
         query_level_start_index) = self._get_encoder_visir_input(query_feats)
        visir_spatial_shapes = query_spatial_shapes[:6]
        visir_level_start_index = query_level_start_index[:6]


        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            query_memory, query_spatial_shapes, query_level_start_index, denoising_class, denoising_bbox_unact)

        visir_memory = query_memory[:,:query_level_start_index[6],:]
        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.query_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3

            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))

        #memory_ff = memory[:,visir_level_start_index[6]:,:]
        #sum_memory_ff = paddle.sum(paddle.sum(memory_ff,axis=2),axis=1)

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        if self.average_query == False:
            topk_score, topk_ind = paddle.topk(
                enc_outputs_class.max(-1), self.num_queries, axis=1)
        else:
            topk_score_vis, topk_ind_vis = paddle.topk(
                enc_outputs_class[:,:visir_level_start_index[3],:].max(-1), self.num_queries / 3, axis=1
            )
            topk_score_ir, topk_ind_ir = paddle.topk(
                enc_outputs_class[:,visir_level_start_index[3]:visir_level_start_index[6],:].max(-1), self.num_queries / 3, axis=1
            )
            topk_ind_ir = topk_ind_ir + visir_level_start_index[3]
            topk_score_fuse, topk_ind_fuse = paddle.topk(
                enc_outputs_class[:,visir_level_start_index[6]:,:].max(-1), self.num_queries / 3, axis=1
            )
            topk_ind_fuse = topk_ind_fuse + visir_level_start_index[6]

            topk_score = paddle.concat([topk_score_vis,topk_score_ir,topk_score_fuse],axis=1)
            topk_ind = paddle.concat([topk_ind_vis,topk_ind_ir,topk_ind_fuse],axis=1)

        # ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # topk10_fuse_highest_score = np.zeros(bs)
        # topk10_visir_highest_socre = np.zeros(bs)
        #
        #
        # #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind <= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]) & (topk_ind <= visir_level_start_index[6]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # topk_ind_mask = paddle.where(
        #     (topk_ind >= visir_level_start_index[6]),
        #     paddle.full_like(topk_ind_mask, 3), topk_ind_mask)
        # #
        #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1,
        #     keepdim=True)], axis=1)
        #
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[5]) & (topk_ind < visir_level_start_index[6])), axis=1,
        #     keepdim=True)], axis=1)
        #
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[6]) & (topk_ind < visir_level_start_index[7])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[7]) & (topk_ind < visir_level_start_index[8])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[8])), axis=1, keepdim=True)],axis=1)
        #
        # #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        # #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        # #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)
        #
        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]
        #
        # mmask = topk_ind_mask == 3
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_fuse_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top100 box
        # for xx in range(bs):
        #     for ii in range(100):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        #### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         else:
        #             cv2.circle(vis_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #             cv2.circle(ir_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score


@register
class Multi_RTDETRTransformer_V3_BA(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer_V3_BA, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size



        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_BA(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_BA(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # value_BA_head
        #self.value_BA_head = MLP(1, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.value_BA_head = None

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            enc_topk_bboxes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            value_BA_head=self.value_BA_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2


        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        #
        # #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        # topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        # topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top100 box
        # for xx in range(bs):
        #     for ii in range(100):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        #### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         else:
        #             cv2.circle(vis_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #             cv2.circle(ir_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score




@register
class Multi_RTDETRTransformer_split(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(Multi_RTDETRTransformer_split, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size



        # backbone feature projection
        #self._build_input_proj_layer(backbone_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer_split(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points)
        self.decoder = TransformerDecoder_split(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
        # (memory, spatial_shapes,
        #  level_start_index) = self._get_encoder_input(feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale,
                                            split = True)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
            self._get_decoder_input(gt_meta,
            visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            self.num_queries,
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            enc_topk_bboxes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # topk_score, topk_ind = paddle.topk(
        #     enc_outputs_class.max(-1), self.num_queries, axis=1)
        topk_score_vis, topk_ind_vis = paddle.topk(
            enc_outputs_class[:, :visir_level_start_index[3], :].max(-1), self.num_queries / 2, axis=1
        )
        topk_score_ir, topk_ind_ir = paddle.topk(
            enc_outputs_class[:, visir_level_start_index[3]:, :].max(-1),
            self.num_queries / 2, axis=1
        )
        topk_ind_ir = topk_ind_ir + visir_level_start_index[3]
        topk_score = paddle.concat([topk_score_vis, topk_score_ir], axis=1)
        topk_ind = paddle.concat([topk_ind_vis, topk_ind_ir], axis=1)



        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,self.num_queries)))
        topk_ind_mask.stop_gradient = True
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2


        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        #
        # #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        # topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        # topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top100 box
        # for xx in range(bs):
        #     for ii in range(100):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/query_bbox/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        #### reference point visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        #
        #   ##plot point
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+2,color_r,-1)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         else:
        #             cv2.circle(vis_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #             cv2.circle(ir_imgs[xx],
        #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                        radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-m3fd/mask_ir_point_0/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score





# @register
# class Multi_RTDETRTransformer_V3_3_3L(nn.Layer):
#     __shared__ = ['num_classes', 'hidden_dim', 'eval_size']
#
#     def __init__(self,
#                  num_classes=80,
#                  hidden_dim=256,
#                  num_queries=300,
#                  position_embed_type='sine',
#                  backbone_feat_channels=[512, 1024, 2048],
#                  feat_strides=[8, 16, 32],
#                  num_levels=3,
#
#                  backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
#                  visir_feat_strides=[8, 16, 32, 8, 16, 32],
#                  num_visir_levels=6,
#
#                  num_decoder_points=4,
#                  nhead=8,
#                  num_decoder_layers=6,
#                  dim_feedforward=1024,
#                  dropout=0.,
#                  activation="relu",
#                  num_denoising=100,
#                  label_noise_ratio=0.5,
#                  box_noise_scale=1.0,
#                  learnt_init_query=True,
#                  eval_size=None,
#                  eval_idx=-1,
#                  eps=1e-2):
#         super(Multi_RTDETRTransformer_V3_3_3L, self).__init__()
#         assert position_embed_type in ['sine', 'learned'], \
#             f'ValueError: position_embed_type not supported {position_embed_type}!'
#         assert len(backbone_feat_channels) <= num_levels
#         assert len(feat_strides) == len(backbone_feat_channels)
#         for _ in range(num_levels - len(feat_strides)):
#             feat_strides.append(feat_strides[-1] * 2)
#
#         self.hidden_dim = hidden_dim
#         self.nhead = nhead
#         self.feat_strides = feat_strides
#         self.visir_feat_strides = visir_feat_strides
#         self.num_levels = num_levels
#         self.num_visir_levels = num_visir_levels
#         self.num_classes = num_classes
#         self.num_queries = num_queries
#         self.eps = eps
#         self.num_decoder_layers = num_decoder_layers
#         self.eval_size = eval_size
#
#         # backbone feature projection
#         #self._build_input_proj_layer(backbone_feat_channels)
#
#         self._build_visir_input_proj_layer(backbone_visir_feat_channels)
#
#         # Transformer module
#         decoder_layer = TransformerDecoderLayer(
#             hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
#             num_decoder_points)
#         self.decoder = TransformerDecoder(hidden_dim, decoder_layer,
#                                           num_decoder_layers, eval_idx)
#
#         # denoising part
#         self.denoising_class_embed = nn.Embedding(
#             num_classes,
#             hidden_dim,
#             weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
#         self.num_denoising = num_denoising
#         self.label_noise_ratio = label_noise_ratio
#         self.box_noise_scale = box_noise_scale
#
#         # decoder embedding
#         self.learnt_init_query = learnt_init_query
#         if learnt_init_query:
#             self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
#
#         # encoder head
#         self.enc_output = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(
#                 hidden_dim,
#                 weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#                 bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
#         self.enc_score_head = nn.Linear(hidden_dim, num_classes)
#         self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
#
#         # decoder head
#         self.dec_score_head = nn.LayerList([
#             nn.Linear(hidden_dim, num_classes)
#             for _ in range(num_decoder_layers)
#         ])
#         self.dec_bbox_head = nn.LayerList([
#             MLP(hidden_dim, hidden_dim, 4, num_layers=3)
#             for _ in range(num_decoder_layers)
#         ])
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         # class and bbox head init
#         bias_cls = bias_init_with_prob(0.01)
#         linear_init_(self.enc_score_head)
#         constant_(self.enc_score_head.bias, bias_cls)
#         constant_(self.enc_bbox_head.layers[-1].weight)
#         constant_(self.enc_bbox_head.layers[-1].bias)
#         for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
#             linear_init_(cls_)
#             constant_(cls_.bias, bias_cls)
#             constant_(reg_.layers[-1].weight)
#             constant_(reg_.layers[-1].bias)
#
#         linear_init_(self.enc_output[0])
#         xavier_uniform_(self.enc_output[0].weight)
#         if self.learnt_init_query:
#             xavier_uniform_(self.tgt_embed.weight)
#         xavier_uniform_(self.query_pos_head.layers[0].weight)
#         xavier_uniform_(self.query_pos_head.layers[1].weight)
#         for l in self.input_proj_visir:
#             xavier_uniform_(l[0].weight)
#
#         # init encoder output anchors and valid_mask
#         if self.eval_size:
#             self.anchors, self.valid_mask = self._generate_anchors()
#
#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         return {'backbone_feat_channels': [i.channels for i in input_shape]}
#
#     def _build_input_proj_layer(self, backbone_feat_channels):
#         self.input_proj = nn.LayerList()
#         for in_channels in backbone_feat_channels:
#             self.input_proj.append(
#                 nn.Sequential(
#                     ('conv', nn.Conv2D(
#                         in_channels,
#                         self.hidden_dim,
#                         kernel_size=1,
#                         bias_attr=False)), ('norm', nn.BatchNorm2D(
#                             self.hidden_dim,
#                             weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#                             bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
#         in_channels = backbone_feat_channels[-1]
#         for _ in range(self.num_levels - len(backbone_feat_channels)):
#             self.input_proj.append(
#                 nn.Sequential(
#                     ('conv', nn.Conv2D(
#                         in_channels,
#                         self.hidden_dim,
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                         bias_attr=False)), ('norm', nn.BatchNorm2D(
#                             self.hidden_dim,
#                             weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#                             bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
#             in_channels = self.hidden_dim
#
#     def _build_visir_input_proj_layer(self, backbone_feat_channels):
#         self.input_proj_visir = nn.LayerList()
#         for in_channels in backbone_feat_channels:
#             self.input_proj_visir.append(
#                 nn.Sequential(
#                     ('conv', nn.Conv2D(
#                         in_channels,
#                         self.hidden_dim,
#                         kernel_size=1,
#                         bias_attr=False)), ('norm', nn.BatchNorm2D(
#                             self.hidden_dim,
#                             weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#                             bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
#         in_channels = backbone_feat_channels[-1]
#         for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
#             self.input_proj_visir.append(
#                 nn.Sequential(
#                     ('conv', nn.Conv2D(
#                         in_channels,
#                         self.hidden_dim,
#                         kernel_size=3,
#                         stride=2,
#                         padding=1,
#                         bias_attr=False)), ('norm', nn.BatchNorm2D(
#                             self.hidden_dim,
#                             weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#                             bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
#             in_channels = self.hidden_dim
#
#     def _get_encoder_input(self, feats):
#         # get projection features
#         proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
#         if self.num_levels > len(proj_feats):
#             len_srcs = len(proj_feats)
#             for i in range(len_srcs, self.num_levels):
#                 if i == len_srcs:
#                     proj_feats.append(self.input_proj[i](feats[-1]))
#                 else:
#                     proj_feats.append(self.input_proj[i](proj_feats[-1]))
#
#         # get encoder inputs
#         feat_flatten = []
#         spatial_shapes = []
#         level_start_index = [0, ]
#         for i, feat in enumerate(proj_feats):
#             _, _, h, w = feat.shape
#             # [b, c, h, w] -> [b, h*w, c]
#             feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
#             # [num_levels, 2]
#             spatial_shapes.append([h, w])
#             # [l], start index of each level
#             level_start_index.append(h * w + level_start_index[-1])
#
#         # [b, l, c]
#         feat_flatten = paddle.concat(feat_flatten, 1)
#         level_start_index.pop()
#         return (feat_flatten, spatial_shapes, level_start_index)
#
#     def _get_encoder_visir_input(self, feats):
#         # get projection features
#         proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
#         if self.num_visir_levels > len(proj_feats):
#             len_srcs = len(proj_feats)
#             for i in range(len_srcs, self.num_visir_levels):
#                 if i == len_srcs:
#                     proj_feats.append(self.input_proj_visir[i](feats[-1]))
#                 else:
#                     proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))
#
#         # get encoder inputs
#         feat_flatten = []
#         spatial_shapes = []
#         level_start_index = [0, ]
#         for i, feat in enumerate(proj_feats):
#             _, _, h, w = feat.shape
#             # [b, c, h, w] -> [b, h*w, c]
#             feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
#             # [num_levels, 2]
#             spatial_shapes.append([h, w])
#             # [l], start index of each level
#             level_start_index.append(h * w + level_start_index[-1])
#
#         # [b, l, c]
#         feat_flatten = paddle.concat(feat_flatten, 1)
#         level_start_index.pop()
#         return (feat_flatten, spatial_shapes, level_start_index)
#
#
#     def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None):
#         # input projection and embedding
#         if self.num_visir_levels == 6:
#             visir_feats = vis_feats + ir_feats
#         elif self.num_visir_levels == 5:
#             visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]
#         # (memory, spatial_shapes,
#         #  level_start_index) = self._get_encoder_input(feats)
#
#         (visir_memory, visir_spatial_shapes,
#          visir_level_start_index) = self._get_encoder_visir_input(visir_feats)
#
#         # prepare denoising training
#         if self.training:
#             denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
#                 get_contrastive_denoising_training_group(gt_meta,
#                                             self.num_classes,
#                                             self.num_queries,
#                                             self.denoising_class_embed.weight,
#                                             self.num_denoising,
#                                             self.label_noise_ratio,
#                                             self.box_noise_scale)
#         else:
#             denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
#
#         target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask = \
#             self._get_decoder_input(gt_meta,
#             visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class, denoising_bbox_unact)
#
#         # decoder
#         out_bboxes, out_logits = self.decoder(
#             gt_meta,
#             target,
#             init_ref_points_unact,
#             visir_memory,
#             visir_spatial_shapes,
#             visir_level_start_index,
#             self.dec_bbox_head,
#             self.dec_score_head,
#             self.query_pos_head,
#             attn_mask=attn_mask,
#             topk_ind_mask = topk_ind_mask)
#         return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
#                 dn_meta)
#
#     def _generate_anchors(self,
#                           spatial_shapes=None,
#                           grid_size=0.05,
#                           dtype="float32"):
#         if spatial_shapes is None:
#             spatial_shapes = [
#                 [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
#                 for s in self.visir_feat_strides
#             ]
#         anchors = []
#         for lvl, (h, w) in enumerate(spatial_shapes):
#             grid_y, grid_x = paddle.meshgrid(
#                 paddle.arange(
#                     end=h, dtype=dtype),
#                 paddle.arange(
#                     end=w, dtype=dtype))
#             grid_xy = paddle.stack([grid_x, grid_y], -1)
#
#             valid_WH = paddle.to_tensor([h, w]).astype(dtype)
#             grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
#             wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
#             anchors.append(
#                 paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))
#
#         anchors = paddle.concat(anchors, 1)
#         valid_mask = ((anchors > self.eps) *
#                       (anchors < 1 - self.eps)).all(-1, keepdim=True)
#         anchors = paddle.log(anchors / (1 - anchors))
#         anchors = paddle.where(valid_mask, anchors,
#                                paddle.to_tensor(float("inf")))
#         return anchors, valid_mask
#
#     def _get_decoder_input(self,
#                            gt_meta,
#                            memory,
#                            spatial_shapes,
#                            visir_level_start_index,
#                            denoising_class=None,
#                            denoising_bbox_unact=None):
#         bs, _, _ = memory.shape
#         topk_ind_mask = None
#         # prepare input for decoder
#         if self.training or self.eval_size is None:
#             anchors, valid_mask = self._generate_anchors(spatial_shapes)
#         else:
#             anchors, valid_mask = self.anchors, self.valid_mask
#         memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
#         output_memory = self.enc_output(memory)
#
#         enc_outputs_class = self.enc_score_head(output_memory)
#         enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
#
#         topk_score, topk_ind = paddle.topk(
#             enc_outputs_class.max(-1), self.num_queries, axis=1)
#
#         ## record topk_fenbu
#         # topk_ind_mask = np.zeros((4,300))
#         # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
#         # topk10_vis_score = np.zeros(bs)
#         # topk10_ir_score = np.zeros(bs)
#         # topk10_vis_highest_score = np.zeros(bs)
#         # topk10_ir_highest_score = np.zeros(bs)
#         # topk10_visir_highest_socre = np.zeros(bs)
#         #
#         # vis_highest_count = np.zeros(bs)
#         # ir_highest_count = np.zeros(bs)
#         # visir_highest_count = np.zeros(bs)
#         # vis_count = np.zeros(bs)
#         # ir_count = np.zeros(bs)
#         #
#         #
#         # for ii in range(bs):
#         #     for iii in range(self.num_queries):
#         #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
#         #             topk_record[ii][0] = topk_record[ii][0] + 1
#         #
#         #             #1 is vis
#         #             topk_ind_mask[ii][iii] = 1
#         #
#         #             if vis_highest_count[ii]<1:
#         #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
#         #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
#         #
#         #             if vis_count[ii]<5:
#         #                 vis_count[ii] = vis_count[ii] + 1
#         #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
#         #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
#         #             topk_record[ii][1] = topk_record[ii][1] + 1
#         #
#         #             # 1 is vis
#         #             topk_ind_mask[ii][iii] = 1
#         #
#         #             if vis_highest_count[ii]<1:
#         #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
#         #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
#         #
#         #             if vis_count[ii] < 5:
#         #                 vis_count[ii] = vis_count[ii] + 1
#         #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
#         #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
#         #             topk_record[ii][2] = topk_record[ii][2] + 1
#         #
#         #             # 2 is ir
#         #             topk_ind_mask[ii][iii] = 2
#         #
#         #             # if vis_highest_count[ii]<1:
#         #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
#         #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
#         #             #
#         #             # if vis_count[ii] < 5:
#         #             #     vis_count[ii] = vis_count[ii] + 1
#         #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
#         #
#         #             if ir_highest_count[ii]<1:
#         #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
#         #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
#         #
#         #
#         #             if ir_count[ii] < 5:
#         #                 ir_count[ii] = ir_count[ii] + 1
#         #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
#         #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
#         #             topk_record[ii][3] = topk_record[ii][3] + 1
#         #
#         #             # 2 is ir
#         #             topk_ind_mask[ii][iii] = 2
#         #
#         #             if ir_highest_count[ii]<1:
#         #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
#         #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
#         #
#         #
#         #             if ir_count[ii] < 5:
#         #                 ir_count[ii] = ir_count[ii] + 1
#         #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
#         #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
#         #         #     topk_record[ii][4] = topk_record[ii][4] + 1
#         #         #
#         #         #     if ir_highest_count[ii]<1:
#         #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
#         #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
#         #         #
#         #         #     if ir_count[ii] < 5:
#         #         #         ir_count[ii] = ir_count[ii] + 1
#         #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
#         #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
#         #         #     topk_record[ii][5] = topk_record[ii][5] + 1
#         #         #
#         #         #     if ir_highest_count[ii]<1:
#         #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
#         #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
#         #         #
#         #         #     if ir_count[ii] < 5:
#         #         #         ir_count[ii] = ir_count[ii] + 1
#         #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
#         #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
#         #             topk_record[ii][4] = topk_record[ii][4] + 1
#         #
#         #             if visir_highest_count[ii] < 1:
#         #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
#         #
#         #
#         #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
#         #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
#         #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
#         #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
#         #
#         #     #vis_highest_score[ii] = topk10_vis_score[0]
#         #
#         #
#         #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
#         #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]
#
#
#         # extract region proposal boxes
#         batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
#         batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
#         topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)
#
#         reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
#                                                   topk_ind)  # unsigmoided.
#         enc_topk_bboxes = F.sigmoid(reference_points_unact)
#
#
#         #### reference point visuilize
#         # vis_imgs = []
#         # ir_imgs = []
#         # for xx in range(bs):
#         #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
#         #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
#         # h,w,_ = vis_imgs[0].shape
#         # real_hw = [w,h]
#         # real_hw = np.array(real_hw)
#         # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
#         #
#         #   ##plot point
#         # radius = 4
#         # color_r = (0,0,255)
#         # color_b = (230,216,173)
#         # color_g = (152,251, 152)
#         #
#         # for xx in range(bs):
#         #     for ii in range(300):
#         #         if topk_ind_mask[xx][ii] == 1:
#         #             if topk_score[xx][ii] > 0:
#         #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
#         #                            radius+2,color_r,-1)
#         #             else:
#         #                 cv2.circle(vis_imgs[xx],
#         #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
#         #                            radius-2, color_b, -1)
#         #         elif topk_ind_mask[xx][ii] == 2:
#         #             if topk_score[xx][ii] > 0:
#         #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
#         #                            radius+2,color_r,-1)
#         #             else:
#         #                 cv2.circle(ir_imgs[xx],
#         #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
#         #                            radius-2, color_b, -1)
#         #         else:
#         #             cv2.circle(vis_imgs[xx],
#         #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
#         #                        radius - 2, color_g, -1)
#         #             cv2.circle(ir_imgs[xx],
#         #                        (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
#         #                        radius - 2, color_g, -1)
#         #
#         # for ii in range(bs):
#         #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-flir/reference_point/'+
#         #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
#         #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-5level-flir/reference_point/'+
#         #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])
#
#
#
#
#
#         if denoising_bbox_unact is not None:
#             reference_points_unact = paddle.concat(
#                 [denoising_bbox_unact, reference_points_unact], 1)
#         if self.training:
#             reference_points_unact = reference_points_unact.detach()
#         enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)
#
#         # extract region features
#         if self.learnt_init_query:
#             target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
#         else:
#             target = paddle.gather_nd(output_memory, topk_ind)
#             if self.training:
#                 target = target.detach()
#         if denoising_class is not None:
#             target = paddle.concat([denoising_class, target], 1)
#
#         return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask

@register
class Multi_RTDETRTransformer_V3_RANK(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,

                 backbone_visir_feat_channels=[256, 256, 256, 256, 256, 256],
                 visir_feat_strides=[8, 16, 32, 8, 16, 32],
                 num_visir_levels=6,

                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 key_aware = False,
                 proj_all = False):
        super(Multi_RTDETRTransformer_V3_RANK, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.visir_feat_strides = visir_feat_strides
        self.num_levels = num_levels
        self.num_visir_levels = num_visir_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_size = eval_size
        self.proj_all = proj_all

        if self.proj_all:
            # backbone feature projection
            self._build_input_proj_layer(backbone_visir_feat_channels)

        self._build_visir_input_proj_layer(backbone_visir_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_visir_levels,
            num_decoder_points,key_aware=key_aware)
        self.decoder = TransformerDecoder_RANK(hidden_dim, decoder_layer,
                                          num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        #query rank layer
        self.rank_aware_content_query = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries, hidden_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0))))
            for _ in range(num_decoder_layers - 1)
        ])

        self.pre_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_decoder_layers - 1)
        ])

        self.post_racq_trans = nn.LayerList([
            copy.deepcopy(nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(num_decoder_layers - 1)
        ])

        # Rank-adaptive Classification Head
        self.rank_adaptive_classhead_emb = nn.LayerList([
            copy.deepcopy(nn.Embedding(num_queries, num_classes, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0))))
            for _ in range(num_decoder_layers)
        ])
        # self.rank_adaptive_classhead_emb = nn.LayerList([
        #     copy.deepcopy(nn.Embedding(num_queries, num_classes))
        #     for _ in range(num_decoder_layers)
        # ])

        self._reset_parameters()


    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        for cls_ in self.post_racq_trans:
            linear_init_(cls_)
        for cls_ in self.pre_racq_trans:
            linear_init_(cls_)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj_visir:
            xavier_uniform_(l[0].weight)

        if self.proj_all:
            for l in self.input_proj:
                xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _build_visir_input_proj_layer(self, backbone_feat_channels):
        self.input_proj_visir = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_visir_levels - len(backbone_feat_channels)):
            self.input_proj_visir.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _get_encoder_visir_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj_visir[i](feat) for i, feat in enumerate(feats)]
        if self.num_visir_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_visir_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj_visir[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj_visir[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)


    def forward(self, feats, vis_feats, ir_feats, pad_mask=None, gt_meta=None,topk_ind_mask=None,topk_score = None):
        # input projection and embedding
        if self.num_visir_levels == 6:
            visir_feats = vis_feats + ir_feats
        elif self.num_visir_levels == 5:
            visir_feats = [vis_feats[0], vis_feats[1], ir_feats[0], ir_feats[1], vis_feats[2]+ir_feats[2]]

        if self.proj_all:
            (memory, spatial_shapes,
             level_start_index) = self._get_encoder_input(visir_feats)

        (visir_memory, visir_spatial_shapes,
         visir_level_start_index) = self._get_encoder_visir_input(visir_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        if self.proj_all:
            target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits , topk_ind_mask, topk_score = \
                self._get_decoder_input(gt_meta,
                memory, spatial_shapes, level_start_index, denoising_class, denoising_bbox_unact)
        else:
            target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score = \
                self._get_decoder_input(gt_meta,
                                        visir_memory, visir_spatial_shapes, visir_level_start_index, denoising_class,
                                        denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            gt_meta,
            target,
            init_ref_points_unact,
            visir_memory,
            visir_spatial_shapes,
            visir_level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.rank_aware_content_query,
            self.rank_adaptive_classhead_emb,
            self.pre_racq_trans,
            self.post_racq_trans,
            attn_mask=attn_mask,
            topk_ind_mask = topk_ind_mask,
            topk_score = topk_score)
        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype="float32"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.visir_feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            lvl = lvl % 3
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           gt_meta,
                           memory,
                           spatial_shapes,
                           visir_level_start_index,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        topk_ind_mask = None
        topk_score = None
        # prepare input for decoder
        if self.training or self.eval_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        topk_score, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)

        ## record topk_fenbu
        topk_ind_mask = paddle.to_tensor(np.zeros((bs,300)))
        topk_ind_mask.stop_gradient = True
        # ##
        # topk_record = np.zeros((bs,len(spatial_shapes) + 2))
        # topk10_vis_score = np.zeros(bs)
        # topk10_ir_score = np.zeros(bs)
        # topk10_vis_highest_score = np.zeros(bs)
        # topk10_ir_highest_score = np.zeros(bs)
        # #topk10_visir_highest_socre = np.zeros(bs)


        #tensor efficent
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 1), topk_ind_mask)  # 将1~10范围内的位置置为1
        topk_ind_mask = paddle.where((topk_ind >= visir_level_start_index[3]), paddle.full_like(topk_ind_mask, 2), topk_ind_mask)  # 将11~20范围内的位置置为2
        # #
        #
        # topk_record = paddle.sum(((topk_ind >= visir_level_start_index[0]) & (topk_ind < visir_level_start_index[1])), axis=1, keepdim=True)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[1]) & (topk_ind < visir_level_start_index[2])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[2]) & (topk_ind < visir_level_start_index[3])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[3]) & (topk_ind < visir_level_start_index[4])), axis=1, keepdim=True)],axis=1)
        # topk_record = paddle.concat([topk_record, paddle.sum(
        #     ((topk_ind >= visir_level_start_index[4]) & (topk_ind < visir_level_start_index[5])), axis=1,
        #     keepdim=True)], axis=1)
        # topk_record = paddle.concat([topk_record,paddle.sum(((topk_ind >= visir_level_start_index[5])), axis=1, keepdim=True)],axis=1)

        #a = paddle.add(topk_record[:,0] , topk_record[:,1])
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:,0] , topk_record[:,1]),axis=1)], axis=1)
        #topk_record = paddle.concat([topk_record, paddle.unsqueeze(paddle.add(topk_record[:, 2] , topk_record[:, 3]),axis=1)], axis = 1)

        # mmask = topk_ind_mask == 1
        # mmask = paddle.cast(mmask,dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask,axis=1),axis=1)
        # for ii in range(bs):
        #     topk10_vis_highest_score[ii] = topk_score[ii][position[ii][0]]
        # #topk10_vis_highest_score = paddle.gather_nd(topk_score,position)
        # mmask = topk_ind_mask == 2
        # mmask = paddle.cast(mmask, dtype='int32')
        # position = paddle.unsqueeze(paddle.argmax(mmask, axis=1), axis=1)
        # for ii in range(bs):
        #     topk10_ir_highest_score[ii] = topk_score[ii][position[ii][0]]


        #topk10_ir_highest_score = paddle.gather_nd(topk_score,paddle.argmax(mmask,axis=1,keepdim=True))




        # topk_record = np.zeros((bs, len(spatial_shapes) + 2))
        # for ii in range(bs):
        #     for iii in range(self.num_queries):
        #         if (topk_ind[ii][iii] >= visir_level_start_index[0] and topk_ind[ii][iii] < visir_level_start_index[1]) :
        #             topk_record[ii][0] = topk_record[ii][0] + 1
        #
        #             #1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii]<5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[1] and topk_ind[ii][iii] < visir_level_start_index[2]) :
        #             topk_record[ii][1] = topk_record[ii][1] + 1
        #
        #             # 1 is vis
        #             topk_ind_mask[ii][iii] = 1
        #
        #             if vis_highest_count[ii]<1:
        #                 topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #                 vis_highest_count[ii] = vis_highest_count[ii] + 1
        #
        #             if vis_count[ii] < 5:
        #                 vis_count[ii] = vis_count[ii] + 1
        #                 topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[2] and topk_ind[ii][iii] < visir_level_start_index[3]) :
        #             topk_record[ii][2] = topk_record[ii][2] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             # if vis_highest_count[ii]<1:
        #             #     topk10_vis_highest_score[ii] = topk_score[ii][iii]
        #             #     vis_highest_count[ii] = vis_highest_count[ii] + 1
        #             #
        #             # if vis_count[ii] < 5:
        #             #     vis_count[ii] = vis_count[ii] + 1
        #             #     topk10_vis_score[ii] = topk10_vis_score[ii] + topk_score[ii][iii]
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif (topk_ind[ii][iii] >= visir_level_start_index[3] and topk_ind[ii][iii] < visir_level_start_index[4]):
        #             topk_record[ii][3] = topk_record[ii][3] + 1
        #
        #             # 2 is ir
        #             topk_ind_mask[ii][iii] = 2
        #
        #             if ir_highest_count[ii]<1:
        #                 topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #                 ir_highest_count[ii] = ir_highest_count[ii] + 1
        #
        #
        #             if ir_count[ii] < 5:
        #                 ir_count[ii] = ir_count[ii] + 1
        #                 topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif (topk_ind[ii][iii] >= visir_level_start_index[4] and topk_ind[ii][iii] < visir_level_start_index[5]):
        #         #     topk_record[ii][4] = topk_record[ii][4] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         # elif topk_ind[ii][iii] >= visir_level_start_index[5]:
        #         #     topk_record[ii][5] = topk_record[ii][5] + 1
        #         #
        #         #     if ir_highest_count[ii]<1:
        #         #         topk10_ir_highest_score[ii] = topk_score[ii][iii]
        #         #         ir_highest_count[ii] = ir_highest_count[ii] + 1
        #         #
        #         #     if ir_count[ii] < 5:
        #         #         ir_count[ii] = ir_count[ii] + 1
        #         #         topk10_ir_score[ii] = topk10_ir_score[ii] + topk_score[ii][iii]
        #         elif topk_ind[ii][iii] >= visir_level_start_index[4]:
        #             topk_record[ii][4] = topk_record[ii][4] + 1
        #
        #             if visir_highest_count[ii] < 1:
        #                 topk10_visir_highest_socre[ii] = topk_score[ii][iii]
        #
        #
        #     # topk_record[ii][6] = topk_record[ii][0] + topk_record[ii][1] + topk_record[ii][2]
        #     # topk_record[ii][7] = topk_record[ii][3] + topk_record[ii][4] + topk_record[ii][5]
        #     topk_record[ii][5] = topk_record[ii][0] + topk_record[ii][1]
        #     topk_record[ii][6] = topk_record[ii][2] + topk_record[ii][3]
        #
        #     #vis_highest_score[ii] = topk10_vis_score[0]
        #
        #
        #     topk10_vis_score[ii] = topk10_vis_score[ii] / vis_count[ii]
        #     topk10_ir_score[ii] = topk10_ir_score[ii] / ir_count[ii]


        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)


        ###reference box visuilize
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw = np.array(real_hw)
        # real_hw = np.tile(real_hw,2)
        # real_reference_point = np.array(enc_topk_bboxes) * real_hw
        #
        # #plot top200 box
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]-real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]+real_reference_point[xx][ii][3]/2)),
        #                           (round(real_reference_point[xx][ii][0]+real_reference_point[xx][ii][2]/2),round(real_reference_point[xx][ii][1]-real_reference_point[xx][ii][3]/2)),color,1)
        #
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > -0.3:
        #                 color = (0,0,255)
        #             else:
        #                 color = (0,255,0)
        #             cv2.rectangle(ir_imgs[xx], (
        #             round(real_reference_point[xx][ii][0] - real_reference_point[xx][ii][2] / 2),
        #             round(real_reference_point[xx][ii][1] + real_reference_point[xx][ii][3] / 2)),
        #                           (round(real_reference_point[xx][ii][0] + real_reference_point[xx][ii][2] / 2),
        #                            round(real_reference_point[xx][ii][1] - real_reference_point[xx][ii][3] / 2)),
        #                           color, 1)
        #
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_box300/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])


        # ### reference point visuilize with box
        # vis_imgs = []
        # ir_imgs = []
        # for xx in range(bs):
        #     vis_imgs.append(cv2.imread(gt_meta['vis_im_file'][xx]))
        #     ir_imgs.append(cv2.imread(gt_meta['ir_im_file'][xx]))
        # h,w,_ = vis_imgs[0].shape
        # real_hw = [w,h]
        # real_hw2 = [w,h,w,h]
        # real_hw = np.array(real_hw)
        # real_hw2 = np.array(real_hw2)
        # real_reference_point = np.array(enc_topk_bboxes[:,:,:2]) * real_hw
        # real_bbox = np.array(enc_topk_bboxes) * real_hw2
        #
        #   ##plot point and box
        # radius = 4
        # color_r = (0,0,255)
        # color_b = (230,216,173)
        # color_g = (152,251, 152)
        #
        # for xx in range(bs):
        #     for ii in range(300):
        #         if topk_ind_mask[xx][ii] == 1:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(vis_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+3,color_r,-1)
        #
        #                 # x = round(real_bbox[xx][[ii][0]-real_bbox[xx][ii][2]/2])
        #                 # x = round(real_bbox[xx][[ii][1]+real_bbox[xx][ii][3]/2])
        #                 # x = round(real_bbox[xx][[ii][0]+real_bbox[xx][ii][2]/2])
        #                 # x= round(real_bbox[xx][[ii][1]-real_bbox[xx][ii][3]/2])
        #
        #                 cv2.rectangle(vis_imgs[xx],(round(real_bbox[xx][ii][0]-real_bbox[xx][ii][2]/2),round(real_bbox[xx][ii][1]+real_bbox[xx][ii][3]/2)),
        #                               (round(real_bbox[xx][ii][0]+real_bbox[xx][ii][2]/2),round(real_bbox[xx][ii][1]-real_bbox[xx][ii][3]/2)),color_r,2)
        #             else:
        #                 cv2.circle(vis_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         elif topk_ind_mask[xx][ii] == 2:
        #             if topk_score[xx][ii] > 0:
        #                 cv2.circle(ir_imgs[xx],(round(real_reference_point[xx][ii][0]),round(real_reference_point[xx][ii][1])),
        #                            radius+3,color_r,-1)
        #
        #                 cv2.rectangle(ir_imgs[xx], (round(real_bbox[xx][ii][0] - real_bbox[xx][ii][2] / 2),
        #                                              round(real_bbox[xx][ii][1] + real_bbox[xx][ii][3] / 2)),
        #                               (round(real_bbox[xx][ii][0] + real_bbox[xx][ii][2] / 2),
        #                                round(real_bbox[xx][ii][1] - real_bbox[xx][ii][3] / 2)), color_r, 2)
        #             else:
        #                 cv2.circle(ir_imgs[xx],
        #                            (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #                            radius-2, color_b, -1)
        #         # else:
        #         #     cv2.circle(vis_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #         #     cv2.circle(ir_imgs[xx],
        #         #                (round(real_reference_point[xx][ii][0]), round(real_reference_point[xx][ii][1])),
        #         #                radius - 2, color_g, -1)
        #
        # for ii in range(bs):
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_reference_point_box/'+
        #                 gt_meta['vis_im_file'][ii].split('/')[-1].split('.')[0]+'_vis.png',vis_imgs[ii])
        #     cv2.imwrite('/home/guojunjie/PycharmProjects/pp_detection/PaddleDetection-develop/output/ms-detrv3-newm3fd-nobug/init_query_reference_point_box/'+
        #                 gt_meta['ir_im_file'][ii].split('/')[-1].split('.')[0]+'_ir.png',ir_imgs[ii])





        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits, topk_ind_mask, topk_score
