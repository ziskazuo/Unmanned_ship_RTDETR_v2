"""
改进的雷达-相机融合注意力机制
根据实际数据集特点进行优化
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .deformable_transformer import MSDeformableAttention_RadarCamera


class ImprovedRadarCameraAttention(MSDeformableAttention_RadarCamera):
    """
    改进的雷达-相机融合注意力机制
    基于您的AirSim数据集特点进行优化
    """
    
    def __init__(self, d_model, n_head, n_levels, n_points):
        super().__init__(d_model, n_head, n_levels, n_points)
        
        # 根据AirSim设置计算相机内参
        self.camera_params = self._initialize_camera_params()
        
        # 学习的相机-雷达对齐参数
        self.alignment_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # 模态权重学习
        self.modality_weights = nn.Parameter(paddle.ones([2]))  # radar, camera
        
    def _initialize_camera_params(self):
        """根据AirSim设置初始化相机参数"""
        # 从settings.json提取的参数
        width, height = 1000, 1000
        fov_degrees = 90
        
        # 计算内参矩阵 (基于FOV)
        fov_rad = np.deg2rad(fov_degrees)
        focal_length = (width / 2.0) / np.tan(fov_rad / 2.0)
        
        # 6个相机的外参 (基于您的数据文件名cam_1到cam_6)
        # 假设它们按60度间隔均匀分布
        camera_poses = []
        for i in range(6):
            yaw = i * 60  # 0, 60, 120, 180, 240, 300度
            camera_poses.append({
                'position': [0, 0, 0],  # 相对于雷达的位置
                'yaw': yaw,
                'pitch': 0,
                'roll': 0
            })
            
        return {
            'intrinsic': paddle.to_tensor([
                [focal_length, 0, width/2],
                [0, focal_length, height/2], 
                [0, 0, 1]
            ], dtype='float32'),
            'poses': camera_poses,
            'width': width,
            'height': height
        }
    
    def radar_to_camera_projection(self, radar_points, camera_idx):
        """
        改进的雷达点到相机投影
        Args:
            radar_points: [bs, num_points, 3] 雷达点坐标
            camera_idx: 相机索引 (0-5)
        """
        bs, num_points, _ = radar_points.shape
        
        # 获取相机外参
        camera_pose = self.camera_params['poses'][camera_idx]
        yaw_rad = np.deg2rad(camera_pose['yaw'])
        
        # 旋转矩阵 (简化的2D旋转，因为您的数据似乎是2D雷达)
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        rotation_matrix = paddle.to_tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype='float32')
        
        # 批量旋转
        rotation_matrix = rotation_matrix.unsqueeze(0).expand([bs, -1, -1])
        rotated_points = paddle.bmm(radar_points, rotation_matrix.transpose([0, 2, 1]))
        
        # 投影到相机平面 (假设z=1的虚拟深度)
        camera_points = rotated_points[:, :, :2]  # 只取x,y
        depth = paddle.ones([bs, num_points, 1])
        homogeneous_points = paddle.concat([camera_points, depth], axis=-1)
        
        # 应用内参矩阵
        intrinsic = self.camera_params['intrinsic'].unsqueeze(0).expand([bs, -1, -1])
        projected = paddle.bmm(homogeneous_points, intrinsic.transpose([0, 2, 1]))
        
        # 归一化到像素坐标并转换到[0,1]范围
        pixel_coords = projected[:, :, :2] / projected[:, :, 2:3]
        normalized_coords = pixel_coords / paddle.to_tensor([
            self.camera_params['width'], self.camera_params['height']
        ]).unsqueeze(0).unsqueeze(0)
        
        # 有效性掩码
        valid_mask = (
            (normalized_coords[:, :, 0] >= 0) & 
            (normalized_coords[:, :, 0] <= 1) &
            (normalized_coords[:, :, 1] >= 0) & 
            (normalized_coords[:, :, 1] <= 1)
        )
        
        return normalized_coords, valid_mask
    
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
                **kwargs):
        """
        改进的前向传播
        """
        bs, Len_q = query.shape[:2]
        
        # 投影值
        radar_value = self.value_proj(radar_value)
        camera_value = self.value_proj(camera_value)
        
        if value_mask is not None:
            value_mask = value_mask.astype(radar_value.dtype).unsqueeze(-1)
            radar_value *= value_mask
            
        radar_value = radar_value.reshape([bs, -1, self.num_heads, self.head_dim])
        
        # 处理多相机数据 (6个相机)
        num_cameras = 6
        camera_value = camera_value.reshape([bs * num_cameras, -1, self.num_heads, self.head_dim])
        
        # 生成采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])
        
        # 雷达采样位置
        radar_sampling_locations = (
            reference_points[:, :, None, :3, None, :2] + 
            sampling_offsets[:, :, :, :3, :, :] / self.num_points * 
            reference_points[:, :, None, :3, None, 2:] * 0.5
        )
        
        # 相机采样位置 - 使用改进的投影方法
        camera_sampling_locations_list = []
        camera_mask_list = []
        
        # 从reference_points提取雷达点坐标
        radar_points = reference_points[:, :, 0, :2]  # [bs, num_queries, 2]
        # 添加虚拟z坐标
        radar_points_3d = paddle.concat([
            radar_points, 
            paddle.zeros([bs, Len_q, 1])
        ], axis=-1)
        
        for cam_idx in range(num_cameras):
            # 投影到当前相机
            projected_coords, valid_mask = self.radar_to_camera_projection(
                radar_points_3d, cam_idx
            )
            
            # 学习的坐标对齐
            aligned_coords = self.alignment_net(projected_coords)
            
            # 生成采样位置
            camera_reference = aligned_coords.unsqueeze(2).expand(
                [-1, -1, self.num_levels // 2, -1]
            ).reshape([bs, Len_q, 1, self.num_levels // 2, 1, 2])
            
            camera_sampling_locations = (
                camera_reference + 
                sampling_offsets[:, :, :, 3:, :, :] / 
                paddle.to_tensor(camera_value_spatial_shapes).flip([1]).reshape(
                    [1, 1, 1, self.num_levels // 2, 1, 2]
                )
            )
            
            camera_sampling_locations_list.append(camera_sampling_locations)
            
            # 扩展掩码维度
            expanded_mask = valid_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(
                [bs, Len_q, self.num_heads, self.num_levels // 2, self.num_points, 1]
            ).expand([-1, -1, -1, -1, -1, 2])
            camera_mask_list.append(expanded_mask)
        
        # 转换为张量
        radar_value_spatial_shapes = paddle.to_tensor(radar_value_spatial_shapes)
        radar_value_level_start_index = paddle.to_tensor(radar_value_level_start_index)
        camera_value_spatial_shapes = paddle.to_tensor(camera_value_spatial_shapes)
        camera_value_level_start_index = paddle.to_tensor(camera_value_level_start_index)
        
        # 应用模态权重
        modality_weights = F.softmax(self.modality_weights, dim=0)
        
        # 调用核心注意力函数
        output = self.ms_deformable_attn_core(
            radar_value * modality_weights[0], 
            camera_value * modality_weights[1],
            radar_value_spatial_shapes, 
            radar_value_level_start_index,
            camera_value_spatial_shapes, 
            camera_value_level_start_index,
            radar_sampling_locations, 
            camera_sampling_locations_list, 
            camera_mask_list, 
            attention_weights
        )
        
        output = self.output_proj(output)
        return output


class LearnableRadarCameraAlignment(nn.Layer):
    """
    可学习的雷达-相机对齐模块
    通过对比学习优化跨模态对齐
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.radar_proj = nn.Linear(feature_dim, feature_dim)
        self.camera_proj = nn.Linear(feature_dim, feature_dim)
        self.temperature = nn.Parameter(paddle.ones([]) * 0.07)
        
    def forward(self, radar_features, camera_features, correspondence_mask=None):
        """
        对比学习的特征对齐
        Args:
            radar_features: [bs, num_queries, feature_dim]
            camera_features: [bs, num_queries, feature_dim] 
            correspondence_mask: [bs, num_queries] 对应关系掩码
        """
        # 特征投影和归一化
        radar_proj = F.normalize(self.radar_proj(radar_features), axis=-1)
        camera_proj = F.normalize(self.camera_proj(camera_features), axis=-1)
        
        # 计算相似性矩阵
        similarity = paddle.bmm(radar_proj, camera_proj.transpose([0, 2, 1])) / self.temperature
        
        # 对比损失
        if correspondence_mask is not None:
            # 正样本：对应的雷达-相机特征对
            positive_sim = paddle.diagonal(similarity, axis1=1, axis2=2)
            
            # 负样本：不对应的特征对
            negative_sim = similarity * (1 - paddle.eye(similarity.shape[1]).unsqueeze(0))
            
            # InfoNCE损失
            pos_exp = paddle.exp(positive_sim)
            neg_exp = paddle.sum(paddle.exp(negative_sim), axis=-1)
            
            alignment_loss = -paddle.mean(paddle.log(pos_exp / (pos_exp + neg_exp)))
            
            return aligned_features, alignment_loss
        
        # 软对齐 (推理时)
        attention_weights = F.softmax(similarity, axis=-1)
        aligned_features = paddle.bmm(attention_weights, camera_proj)
        
        return aligned_features, None