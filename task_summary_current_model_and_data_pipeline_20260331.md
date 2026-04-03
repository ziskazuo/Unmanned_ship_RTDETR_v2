# 当前模型与数据处理全链路说明（用于外部排查）

更新时间：2026-03-31  
项目路径：`/data1/zuokun/code/USV/Unmanned_ship_RTDETR`

---

## 1. 文档目的

这份文档用于给外部同事快速建立“当前代码真实在跑什么”的完整认知，覆盖：

1. 当前实际生效的模型结构（不是历史方案）
2. 数据构建流程（离线）
3. 训练/验证/测试的数据读取与预处理流程（在线）
4. 监督信号与损失设计
5. 验证指标与最优权重保存逻辑
6. 当前实现的关键假设与潜在风险点

---

## 2. 当前正在使用的训练配置与命令

当前训练进程实际命令（含覆盖项）：

```bash
/data1/zuokun/vene/Unmanned_ship_RTDETR/bin/python -m paddle.distributed.launch \
  --gpus 0,1,2 \
  tools/train.py \
  -c configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_16e_bs2x4_fp32_test400_onlineval_4km_super4_r1536_c1024x512.yml \
  --eval \
  -o TrainReader.batch_size=1
```

说明：

1. 主配置文件是 `...r1536_c1024x512.yml`
2. 训练中开启了 `--eval`（边训练边验证）
3. 实际 batch 被命令行覆盖为 `TrainReader.batch_size=1`
4. 当前命令未使用 `--amp`，即纯 FP32 训练

---

## 3. 任务定义与类别体系

### 3.1 检测目标

雷达 BEV 上做旋转框检测（RBox），同时利用 4 路 RGB 相机特征增强分类。

### 3.2 当前类别

当前按 4 大类训练/评估：

1. CargoShip
2. CruiseShip
3. FishingVessel
4. RecreationalBoat

对应细分类映射（构建数据时执行）：

1. CargoShip: Containership, libertyship, smallcargo, suppliervessel
2. CruiseShip: queenmarry, ramonasteam
3. FishingVessel: fishingboat
4. RecreationalBoat: Yacht, Sailboat, HouseBoat, Motorboat, Boataaa, CoastGuard

---

## 4. 当前模型结构（真实生效版）

## 4.1 总体结构

当前是“雷达主线 + 相机 RouteROI 分类增强”：

1. 雷达 backbone + neck + transformer decoder 负责候选框/框回归/角度
2. 相机分支不再进入 transformer 的 proposal 主链
3. 相机在 head 侧通过 RouteROI 参与路由、ROI 提特征、分类增强

对应组件：

1. Architecture: `RadarCamera_RouteROI_DETR`
2. Radar backbone: `ResNetRadarP2`
3. Camera backbone: `ResNet`
4. Radar neck: `RadarHybridEncoderP2`
5. Camera neck: `HybridEncoder`
6. Transformer: `RTDETRTransformer_Rotate_RouteROI`
7. Head: `DINOHead_Rotate_RouteROI`
8. PostProcess: `DETRPostProcess_Rotate`

---

## 4.2 输入与特征分辨率

按当前配置：

1. 雷达输入分辨率：`1536 x 1536`
2. 相机输入分辨率：`1024 x 512`（W x H，内部 shape 为 H x W = 512 x 1024）
3. 相机数量：4 路（Back, Front, Left, Right）

主干/neck后的核心尺度：

1. 雷达分支最终送入 transformer 的 3 层特征 stride: `[4, 8, 16]`
2. 相机分支用于 ROIAlign 的 3 层特征 stride: `[8, 16, 32]`

---

## 4.3 雷达主线（proposal、框、角度）

`RTDETRTransformer_Rotate_RouteROI` 仍基于旋转 RT-DETR 主体，关键点：

1. query 数量：300
2. decoder 层数：6
3. proposal topk 按层选取：`[180, 90, 30]`
4. 角度离散 bin：`0~90` 共 91 个 bin（`angle_max=90`）
5. 输出包含 `last_query_feat`（供 RouteROI head 使用）

雷达主线职责：

1. 提供最终雷达框（cx, cy, w, h）和角度
2. 提供雷达分类 logits（作为融合前基线）
3. 提供 query 特征给相机分支做 route/ROI/class enhancement

---

## 4.4 RouteROI 相机增强分支

`DINOHead_Rotate_RouteROI` 中相机增强逻辑如下。

### A. 几何投影先验

1. 把雷达预测 rbox 转成四角点
2. 从像素坐标转回米制坐标（依赖 `lidar_range`, `ppi_res`）
3. 结合每路 `camera_intrinsics` + `camera_extrinsics` 投影到各相机
4. 得到：
   1. 每路几何 2D box（`geom_boxes`）
   2. 每路几何可见性（`geom_visible`）

### B. 路由（Route）

输入特征拼接：

1. `last_query_feat`
2. 雷达框与角度
3. 几何可见性（4维）
4. 几何框归一化后展开（4路 x 4坐标）

输出：

1. `route_primary_head`: 5 类（Back/Front/Left/Right/None）
2. `visible_head`: 4 维可见性 logits

路由概率计算：

1. 使用 `route_primary_logits[:, :, :4] + visible_logits - invalid_bias`
2. 几何不可见相机不是硬删除，而是加惩罚（soft prior）
3. 与 `none` 拼接后 softmax
4. 推理时当前实现是 Top-1 路由（`argmax`），不是 Top-2

### C. 两阶段 ROI 提特征

1. 第一次：用几何粗框做 ROIAlign（3层相机特征拼接）
2. 通过 `coarse_box_embed` 预测 box delta，得到 refined box
3. 第二次：用 refined box 再做 ROIAlign，得到 refined ROI feature

### D. 分类融合

1. 每路 ROI 特征先过 `cam_cls_head` 得相机分类 logits（用于辅助监督）
2. 根据路由选中的相机，取对应 ROI 特征
3. `fuse_cls_head` 产生 `fused_delta`
4. 最终分类：`fused_logits = radar_logits + fused_delta * use_camera`
5. 如果选中 `none` 或路由相机几何不可见，回退到纯雷达分类

---

## 4.5 当前实现中“已做”和“未做”

已做：

1. 明确 5 类路由（含 none）
2. 可见性分支监督（multi-label）
3. 几何先验软约束
4. 两阶段 ROI（coarse -> refine -> re-ROI）
5. camera-only 分类辅助头
6. 融合分类头

未做（或尚未启用）：

1. Top-2 路由推理逻辑（当前仅 Top-1）
2. camera 分支 teacher forcing 调度（当前无显式 GT->Pred ROI 退火）
3. 用 `gt_camera_corners_2d` 做投影监督（当前主用 `gt_camera_box_2d`）

---

## 5. 监督与损失（当前生效）

## 5.1 基础检测损失（雷达）

基础损失来自 `DINOLoss_Rotate`：

1. 分类损失 `loss_class`（DINO 主分类）
2. 框回归 `loss_bbox`
3. 旋转 IoU 相关项 `loss_piou`
4. 角度损失 `loss_angle`

角度损失采用动态掩码：

1. 当 GT 框短边 < `angle_ignore_short_side`（当前=2.0 px）时，不计 angle loss
2. 小目标仍参与 cls/bbox，不直接删除

## 5.2 RouteROI 新增损失

在 `DINOLoss_Rotate_RouteROI` 中增加：

1. `loss_route_primary`：5 类路由 CE
2. `loss_visible`：4 维可见性 BCE
3. `loss_cam_class`：相机分支分类 CE
4. `loss_proj_l1`：refined 2D box 的 L1（归一化后）
5. `loss_proj_iou`：refined 2D box 的 IoU loss

## 5.3 匹配与正样本来源

1. 仍使用同一套 Hungarian 匹配（以主检测任务为核心）
2. RouteROI 额外损失只在 matched positive 上计算
3. `primary_camera == none` 或无相机框时，会跳过对应 camera/proj 子损失

## 5.4 损失权重（当前配置）

1. class: 1
2. bbox: 5
3. giou(piou): 2
4. route_primary: 1
5. visible: 1
6. cam_class: 1
7. proj_l1: 2
8. proj_iou: 2

---

## 6. 数据构建流程（离线）

构建脚本：`tools/build_sealand_single_tower_dataset.py`

## 6.1 输入源

每帧核心输入：

1. 雷达点云：`radar_pcd/radar_{frame}.pcd`
2. 雷达标签：`gt_filter/gt_{frame}_tosensor_filter.json`
3. RGB 2D标签：`gt_filter_only_yaw/gt_{frame}_tosensor_filter.json`
4. 相机图像：4 路 `cams/Cam*/rgb/{frame}.png`

## 6.2 雷达训练图生成

1. 将 pcd 点投到笛卡尔 BEV 灰度图
2. 基于 `range_max` 与 `resolution` 建立米-像素映射
3. 使用 `log1p + percentile clip` 做强度归一化

当前目标链路已在 4km 范围使用（`range_max=4000m`）。

## 6.3 雷达旋转框标签生成

优先来源：

1. `radar_proj.corners_3d`（优先）
2. `bev_rot_only_yaw`（仅 fallback）

处理逻辑：

1. 由 corners 生成 raw polygon
2. 先做图像边界裁剪（保留边界目标）
3. 对图内可见 polygon 重拟合最小外接旋转矩形
4. 将训练 polygon 回算到米制，导出 `rbox_xywhr`

这条链路实现了“边界目标保留、按图内可见部分重标”。

## 6.4 相机监督生成

每个目标在 4 路相机上导出：

1. `gt_primary_camera`（0~3，或 4=None）
2. `gt_visible_cameras`（4维 0/1）
3. `gt_camera_box_2d`（4路 xyxy）
4. `gt_camera_poly_area`（4路）
5. `gt_has_camera_box`（4维 0/1）

主相机选择规则：

1. 优先用 `corners_2d` 裁剪后的可见 polygon 面积
2. 若 polygon 不可用，再退化到 box 面积
3. 边界截断目标仍参与可见性与主相机选择

## 6.5 相机顺序规范（非常关键）

全链路 canonical 顺序固定为：

1. 0 = Back
2. 1 = Front
3. 2 = Left
4. 3 = Right
5. 4 = None（仅 route primary 目标类中使用）

COCO `images.camera_names` 与 `camera_im_file` 均按此顺序输出。

---

## 7. 训练时在线数据流（DataLoader -> 模型）

## 7.1 Dataset 解析（CameraRadar_COCODataSet）

从 COCO 解析：

1. 雷达图路径、4路相机路径
2. 检测 GT（bbox/poly/class）
3. RouteROI 监督（primary/visible/camera_box/has_box）

并对相机路径做 canonical 重排（避免顺序漂移）。

## 7.2 Decode 与标定加载（RadarCamera_Decode）

每个样本：

1. 读雷达图与4路相机图，解码为 RGB
2. 写入 `im_shape`、`scale_factor`
3. 加载标定：
   1. 先尝试 `gt_meta`
   2. 若没有则回退 `opv2v_yaml`

当前数据抽样显示：多数走 opv2v 回退路径（`gt_meta` 缺失）。

回退路径关键默认值：

1. `lidar_range = 4000`
2. `projection_plane_height = -6.0`
3. `camera_img_size = [1024, 512]`

## 7.3 Resize（RadarCamera_RResize）

当前配置：

1. 雷达 resize 到 `1536 x 1536`
2. 相机 resize 到 `512 x 1024`（H x W）
3. `keep_ratio=False`

同步缩放：

1. `camera_intrinsics`（fx/fy/cx/cy）
2. `camera_img_size`
3. `gt_camera_box_2d`
4. `gt_camera_corners_2d`（若字段存在）
5. `gt_camera_poly_area`

## 7.4 Normalize + Permute

归一化（RadarCamera_NormalizeImageSeparate）：

1. 雷达：仅 `/255`，不做 mean/std（norm_type=none）
2. 相机：ImageNet mean/std

排布（RadarCamera_Permute）：

1. 雷达：HWC -> CHW
2. 每路相机：HWC -> CHW

进入 backbone 前，相机张量在 backbone 内被 reshape 为 `[B*4, C, H, W]`。

---

## 8. 评估与最优模型保存逻辑

## 8.1 指标

当前 metric：`RBOX_COCO_STYLE`

阈值集合（当前配置）：

1. IoU: 0.50, 0.55, ..., 0.90

输出指标：

1. `bbox_AP50`
2. `bbox_AP75`
3. `bbox_mAP50_90`
4. `bbox_ship_AP50`（一分类，class-agnostic）
5. `bbox_ship_AP75`
6. `bbox_ship_mAP50_90`

说明：一分类指标是在同一次预测结果上做标签折叠统计，不需要再次前向推理。

## 8.2 最优权重选择

Checkpointer 逻辑：

1. 主排序键：`bbox`（即 AP50）
2. 若 AP50 差距小于 `best_model_ap50_tie_margin`（当前 0.001），再看 tie-break
3. tie-break 键：`bbox_mAP50_90`

即：AP50 接近时，优先 mAP50:90 更高的模型。

## 8.3 评估过程记录

启用：

1. `record_eval_history: True`
2. `eval_history_file: eval_history.jsonl`

每次 eval 会记录：

1. epoch
2. fps
3. 多指标结果
4. 当前 best AP50 与 best tie-break

---

## 9. 当前数据规模（实际文件统计）

数据目录：`prepared/sealand_single_tower_4km_super4_1536_route_roi`

统计：

1. `train_coco.json`: 17600 images, 92261 annotations, 4 classes
2. `test_onlineval_random400_seed20260327.json`: 400 images, 1822 annotations, 4 classes
3. `test_coco.json`: 7200 images, 34525 annotations, 4 classes
4. `valid_coco.json`: 1600 images, 9903 annotations, 4 classes（文件存在，但当前主训练配置未使用）

---

## 10. 关键假设与风险点（供外部排查优先关注）

1. 相机投影平面高度依赖 `projection_plane_height`，当前默认 `-6.0`，如果真实安装/坐标定义偏差，会直接影响 RouteROI 质量。
2. 标定加载优先 `gt_meta`，否则回退 `opv2v`；不同来源参数定义若不一致，会造成训练样本间投影噪声。
3. Route 目前是 Top-1，不是 Top-2，对边界跨相机场景容错更弱。
4. 相机投影监督当前核心是 `gt_camera_box_2d`，未利用 `gt_camera_corners_2d` 进行更细几何约束。
5. 当前没有 camera 分支 teacher forcing 退火，早期 route/proj 可能受雷达预测噪声影响较大。
6. Eval 集使用 `test` 中随机 400 子集，适合快速选权重，但统计稳定性低于全量验证。
7. 配置声明 TrainReader batch_size=2，但运行命令覆盖为 1，复现实验时必须注意命令行覆盖项。

---

## 11. 一句话总结当前方案

当前方案已从“decoder 内四路平均融合”切换为“雷达主检测 + 相机 RouteROI 分类增强”，数据侧已接入主相机/可见性/2D框监督并统一相机 ID 顺序；训练评估支持四分类与一分类同时在线记录，最优模型按 AP50 主排序、mAP50:90 近似 tie-break 保存。

