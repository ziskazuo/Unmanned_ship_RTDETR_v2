# 当前模型架构、数据流向、数据来源与监督设置（代码核对版）

更新时间：2026-04-25 10：08  
仓库：`/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2`

---

## 0. 文档范围与“当前”口径

本文档基于仓库内**当前默认主配置**进行逐代码核对，口径如下：

- 主配置：`configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8.yml`
- 默认训练入口：`tools/train_route_roi_onlineval.sh`（默认就指向上述配置）
- 多卡等待调度默认入口：`tools/wait_and_launch_sealand_multigpu.py`（默认也指向上述配置）

若后续改用 `4km`、`ship1` 或其它配置文件，本文档中的尺寸、数据源与监督字段会相应变化。

---

## 1. 当前生效配置总览

### 1.1 配置继承链

当前主配置继承：

1. `configs/datasets/sealand_single_tower_radarcamera_2km_super4_1536_route_roi_hbb_min8.yml`
2. `configs/runtime_radar.yml`
3. `configs/rtdetr/_base_/sealand_radardetr_reader_r1536_c1024x512_super4.yml`
4. `configs/rtdetr/_base_/radardetr_r50vd_route_roi_p2.yml`

### 1.2 关键训练与评估参数（当前值）

- 训练轮数：`epoch = 30`
- 训练 batch：`TrainReader.batch_size = 1`
- 验证 batch：`EvalReader.batch_size = 1`
- 优化器：`AdamW`
- 基础学习率：`4e-5`
- 调度：`CosineDecay + LinearWarmup(2000 steps)`
- 梯度裁剪：`clip_grad_by_norm = 0.1`
- `find_unused_parameters = True`
- `convert_sync_bn = False`
- 指标类型：`RBOX_COCO_STYLE`
- IoU 阈值：`0.50:0.05:0.90`
- 开启 ship-only 统计：`enable_ship_only_eval = True`
- best 规则：主指标 AP50，若接近（margin=`0.005`）则用 `mAP50:90` 打平
- `eval_size = [1536, 1536]`

### 1.3 模型主干配置参数（当前值）

- 架构：`RadarCamera_RouteROI_DETR`
- Radar Backbone：`ResNetRadarP2`
- Camera Backbone：`ResNet`
- Radar Neck：`RadarHybridEncoderP2`
- Camera Neck：`HybridEncoder`
- Transformer：`RTDETRTransformer_Rotate_RouteROI`
- Head：`DINOHead_Rotate_RouteROI`
- PostProcess：`DETRPostProcess_Rotate`
- `num_queries = 300`
- decoder 层数：`6`
- Radar 特征层 stride：`[4, 8, 16]`
- proposal 分层 topk：`[180, 90, 30]`
- proposal learned rerank：开启
- 类别数：`4`（CargoShip / CruiseShip / FishingVessel / RecreationalBoat）

---

## 2. 当前模型架构（模块与职责）

## 2.1 总体结构

当前是“雷达主检测 + 相机 RouteROI 分类增强”路线：

- 雷达分支负责候选与主检测（框、角度、基础类别）
- 相机分支不进入主 proposal 生成链
- 相机在 head 端通过几何投影、路由、ROI 两阶段提特征后做分类增益融合

## 2.2 Radar 分支

输入：`radar_image`，分辨率 `1536x1536`

- `ResNetRadarP2` 输出 5 级特征（含 stem）：
  - stem stride2，后续 stage stride4/8/16/32
- `RadarHybridEncoderP2` 将其融合为 3 层检测特征：
  - 输出 stride：`4, 8, 16`
  - 输出通道统一为 `256`

## 2.3 Camera 分支

输入：`camera_image`，每样本 4 路相机图像

- 预处理后单样本相机张量为 `4 x 3 x 512 x 1024`
- `ResNet` 前向时会先把 `B x 4` 展平到 batch 维，即 `(B*4)`
- `ResNet(return_idx=[1,2,3]) + HybridEncoder` 输出 3 层相机特征：
  - stride：`8, 16, 32`
  - 每层通道：`256`

## 2.4 Transformer（旋转 RT-DETR 主干）

- `RTDETRTransformer_Rotate_RouteROI` 基于旋转 RT-DETR
- `num_queries = 300`
- angle 离散：`angle_max = 90`，对应 `91` 个角度 bin
- proposal 不是简单 global topk，而是：
  - 先按层候选池（`[180,90,30]`）
  - 再叠加 level bias、小框 bonus、learned rerank 分数
  - 最终取 top-queries
- decoder 最终额外返回 `last_query_feat`，供 RouteROI head 使用

## 2.5 RouteROI Head（`DINOHead_Rotate_RouteROI`）

### A. 几何投影

- 将雷达预测 `(cx,cy,w,h,angle)` 转成四角点
- 像素坐标转米制（使用 `lidar_range` 与雷达分辨率）
- 使用 `camera_intrinsics/camera_extrinsics` 投影到 4 路相机
- 得到：
  - `geom_boxes`（每 query 每相机的几何 2D 框）
  - `geom_visible`（几何可见性）

### B. 路由与可见性

Route 输入拼接项：

- `last_query_feat`
- 雷达框+角度
- 几何可见性（4 维）
- 几何框归一化后展平（4x4）

Route 输出：

- `route_primary_logits`：5 类（Back/Front/Left/Right/None）
- `visible_logits`：4 路可见性

推理路由概率：

- 对前 4 路相机 logits 加上 `visible_logits`
- 对几何不可见相机加惩罚（soft prior，不是硬屏蔽）
- 与 none 类拼接后 softmax
- 当前推理使用 `argmax`，即 Top-1 路由

### C. 两阶段 ROI 特征提取

- 第 1 阶段：对几何粗框做多层 ROIAlign
- 预测 box delta 得到 refined box
- 第 2 阶段：对 refined box 再 ROIAlign
- 产出每路相机 ROI 特征与 refined box

### D. 分类融合

- 每路 ROI 特征先经 `cam_cls_head`（用于辅助监督）
- 按路由选中相机抽取 ROI 特征
- 与 query 特征拼接后经 `fuse_cls_head` 产生 `fused_delta`
- 最终分类：`fused_logits = radar_logits + fused_delta * use_camera`
- 若选中 none 或相机几何不可见，则回退雷达分类

## 2.6 后处理输出

`DETRPostProcess_Rotate` 将 `(x,y,w,h,angle)` 转为 8 点多边形，最终输出格式：

- `[label, score, x1,y1,x2,y2,x3,y3,x4,y4]`
- 默认 `num_top_queries = 100`

---

## 3. 当前数据流向（端到端）

## 3.1 数据来源到 prepared 资产

按当前 2km prepared 数据集样本索引（`train_samples.jsonl`）可追溯到原始来源：

- 雷达原始点云：`.../radar_pcd/radar_XXXXXX.pcd`
- 雷达/旋转框主标注来源：`.../gt_filter/gt_XXXXXX_tosensor_filter.json`
- RGB 2D 监督来源：`.../gt_filter_only_yaw/gt_XXXXXX_tosensor_filter.json`
- 参考标注：`gt_sensor` / `gt` / `opv2v_yaml`
- 生成后的训练雷达图写入 prepared 下的 `Train|Valid|Test/radar_bev/...`

构建逻辑见：`tools/build_sealand_single_tower_dataset.py`。

## 3.2 COCO 读取层

当前数据集类：`CameraRadar_COCODataSet`

- 读取 `radar_im_file` 与 `camera_im_file`
- 相机路径会尝试重排到固定顺序：`Back, Front, Left, Right`
- 当前 train/eval data_fields：
  - `image, gt_bbox, gt_class, is_crowd, gt_poly`
  - `gt_primary_camera, gt_visible_cameras, gt_camera_box_2d, gt_has_camera_box`
- 未在当前 data_fields 中启用：`gt_camera_poly_area`

额外标签处理：

- `min_gt_rbox_edge = 8.0`：对 `segmentation` 执行最小边长约束（避免过小退化框）
- 再由 `Poly2RBox(rbox_type='oc')` 生成 `gt_rbox`

## 3.3 预处理链路（TrainReader）

当前顺序：

1. `RadarCamera_Decode`
2. `Poly2Array`
3. `RadarCamera_RResize`（雷达到 `1536x1536`，相机到 `512x1024`）
4. `Poly2RBox`
5. `RadarCamera_NormalizeImageSeparate`
6. `RadarCamera_Permute`

关键点：

- 雷达与相机分辨率独立缩放
- 缩放时同步更新 `camera_img_size` 与 `camera_intrinsics`
- 同步缩放 `gt_camera_box_2d`

## 3.4 标定与投影元信息注入

`RadarCamera_Decode` 会尝试加载并注入：

- `lidar_range`
- `ppi_res`
- `camera_intrinsics`
- `camera_extrinsics`
- `camera_img_size`
- `height_range`
- `projection_plane_height`（若数据中提供）

若部分元信息缺失，RouteROI 投影默认平面高度回退到 `-6.0`。

## 3.5 训练/验证/测试读取差异

- Train/Eval：读取 GT 与 RouteROI 监督字段
- Test：`data_fields=['image']`，只做前向推理，不读 GT
- 训练时若传 `--eval`，Trainer 会在每轮后跑 `EvalDataset`（当前配置是 `valid_coco.json`）

---

## 4. 训练/验证/测试数据来源（当前配置）

数据集根目录：

- `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8`

### 4.1 Split 文件

- Train：`train_coco.json`
- Valid：`valid_coco.json`
- Test：`test_coco.json`

### 4.2 规模统计（当前文件实测）

- Train：`17600` 图，`85795` 标注，空标注图 `15`
- Valid：`1600` 图，`9123` 标注，空标注图 `0`
- Test：`7200` 图，`30148` 标注，空标注图 `0`

### 4.3 场景与塔位覆盖（按 file_name 统计）

- Train：8 个场景（`T01~T08`），22 个 `scene/tower`
- Valid：1 个场景（`V01`），2 个 `scene/tower`
- Test：3 个场景（`S01,S02,S03`），9 个 `scene/tower`

### 4.4 类别分布（标注数）

Train：

- CargoShip：29274
- CruiseShip：12002
- FishingVessel：26639
- RecreationalBoat：17880

Valid：

- CargoShip：2321
- CruiseShip：2578
- FishingVessel：1676
- RecreationalBoat：2548

Test：

- CargoShip：12169
- CruiseShip：6116
- FishingVessel：7826
- RecreationalBoat：4037

### 4.5 RouteROI 监督覆盖（Train）

字段覆盖率（在 `annotations` 中存在该键）：

- `gt_primary_camera`：100%
- `gt_visible_cameras`：100%
- `gt_camera_box_2d`：100%
- `gt_has_camera_box`：100%

主相机分布（Train）：

- Back(0)：21414
- Front(1)：30175
- Left(2)：19125
- Right(3)：13971
- None(4)：1110

可见相机数量（每个 GT）：

- 可见 0 路：1110
- 可见 1 路：84685

结论：当前数据基本是单主视角监督，少量 GT 为 none/不可见。

---

## 5. 当前监督设置与损失

## 5.1 主检测监督（雷达空间）

由 `DINOLoss_Rotate` 提供：

- `loss_class`
- `loss_bbox`（L1）
- `loss_piou`（ProbIoU）
- `loss_angle`（角度分布损失）
- 以及 aux 与 dn 分支对应损失

角度监督掩码：

- 当匹配 GT 的短边 `< angle_ignore_short_side(=2.0px)` 时，角度损失被掩蔽
- 这类目标仍参与分类与框回归

## 5.2 RouteROI 增强监督

由 `DINOLoss_Rotate_RouteROI` 额外增加：

- `loss_route_primary`：5 类主相机路由 CE
- `loss_visible`：4 路可见性 BCE
- `loss_cam_class`：相机分支分类 CE
- `loss_proj_l1`：refined 2D 框归一化 L1
- `loss_proj_iou`：refined 2D 框 IoU 损失

对应 GT 字段：

- `gt_primary_camera`
- `gt_visible_cameras`
- `gt_camera_box_2d`
- `gt_has_camera_box`

## 5.3 匹配机制

- 检测主任务匹配：`HungarianMatcher_Rotate`
- 成本项：分类 + bbox + piou + angle dfl
- RouteROI 额外损失只在**已匹配 query**上计算
- `cam_class / proj_*` 进一步要求 `primary_camera < 4` 且该相机 `gt_has_camera_box=1`

---

## 6. 评估与 best 保存策略（当前实现）

## 6.1 指标

`RBoxCocoStyleMetric` 输出：

- `bbox` / `bbox_AP50`
- `bbox_AP75`
- `bbox_mAP50_90`
- 以及 ship-only 对应指标（若开启）

## 6.2 训练中验证集来源

- 训练 `--eval` 时使用 `EvalDataset`
- 当前主配置 `EvalDataset.anno_path = valid_coco.json`

## 6.3 best 模型保存

- 主排序：`bbox(AP50)`
- 若 AP50 差值在 `best_model_ap50_tie_margin=0.005` 内，使用 `bbox_mAP50_90` 决胜
- 支持记录 `eval_history.jsonl`

---

## 7. 已确认事实与注意点

1. 当前默认主线已经切到 `2km + r1536/c1024x512 + hbb_min8`，不是旧的 `4km/960` 主配置。
2. RouteROI 训练监督链路完整打通，且数据中相应字段覆盖率为 100%。
3. 当前 Train split 存在少量空标注图（15 张），配置里 `allow_empty=true`，会进入训练。
4. RouteROI 推理目前是 Top-1 相机路由，不是 Top-2。
5. 若标定元信息缺失，投影平面高度会回退到默认值（`-6.0`），这会影响几何投影精度。

