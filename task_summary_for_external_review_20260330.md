# 项目任务与当前方案说明

更新时间：2026-03-30

## 1. 任务背景

这是一个多模态船只检测任务，目标是在单塔位场景下，融合：

- 雷达 BEV 图
- 四路非重叠 RGB 相机

完成船只检测与分类。

当前业务设定的核心特点：

- 相机是 `Front / Back / Left / Right` 四路，视场基本不重叠
- 小目标很多，极小目标占比高
- 雷达负责主定位更稳
- 相机更适合补分类和外观信息

因此，我们当前主线不是“完全对称的雷达相机融合”，而是：

**雷达主检测 + 相机后端 RouteROI 分类增强**

## 2. 数据与监督

### 2.1 数据来源

当前训练/测试使用的 prepared 数据集路径：

- `prepared/sealand_single_tower_4km_super4_960/train_coco.json`
- `prepared/sealand_single_tower_4km_super4_960/test_coco.json`

当前在线验证使用的是固定的 `400` 张测试子集：

- `prepared/sealand_single_tower_4km_super4_960/test_onlineval_random400_seed20260327.json`

### 2.2 数据规模

当前训练链路中实际使用的是：

- 训练集：`17600` 张
- 完整测试集：`7200` 张
- 在线验证：测试集随机 `400` 张

### 2.3 当前监督信号

当前模型实际使用的监督包括：

- 雷达主检测：
  - `gt_bbox`
  - `gt_poly`
  - `gt_class`
- 相机辅助监督：
  - `gt_primary_camera`
  - `gt_visible_cameras`
  - `gt_camera_box_2d`
  - `gt_camera_poly_area`

已经删除、不再参与训练的监督：

- `gt_camera_corners_2d`

说明：

- 当前相机 2D 监督只保留 `camera_box_2d`
- `camera_corners_2d` 已从模型 loss 中移除

### 2.4 GT 来源现状

当前 prepared 训练监督主要来源于：

- `gt_filter/*_tosensor_filter.json`

说明：

- 原始 `gt_sensor` 与 `gt_filter` 之间存在差异
- 近期我们发现 RGB 2D GT 本身仍存在部分问题，用户已经在：
  - `/data1/ZK/Dataset/USV_DATASET_v4/StressTest/H01/gt_sensor_only_yaw/`
  单独修过一版
- 目前这件事还未完全反映到当前正式训练链路

## 3. 当前模型方案

当前正式方案不是“重构融合范式”的版本，而是：

**在 RouteROI/P2 路线基础上，优先修复 proposal / 小目标主检测 / 训练策略问题，不改融合主范式**

### 3.1 总体结构

当前结构主线：

`radar backbone -> radar neck -> radar transformer proposal/decoder -> camera router -> ROIAlign -> camera classification enhancement`

也就是说：

- 雷达负责主 proposal 和主框回归
- 相机分支仍然是 late fusion
- 相机目前不参与主 proposal 生成

### 3.2 当前雷达主检测结构

当前雷达 backbone/neck 不是旧版 `[8,16,32]`，而是 P2 小目标版：

- backbone：`ResNetRadarP2`
  - 返回 `stride2 / 4 / 8 / 16 / 32`
- neck：`RadarHybridEncoderP2`
  - 最终输出三层检测特征：`[4, 8, 16]`
  - `stride2` 只作为细节注入，不直接作为独立 proposal level

当前关键设计：

- `stride16` 是主编码层
- `stride32` 只保留为深语义输入，不再作为主编码层
- `P2/P3` 走轻量增强，不做重型全局 attention

### 3.3 当前 proposal 机制

当前 proposal 不是纯手工 top-k，而是：

- 先按层保留候选池：`[180, 90, 30]`
- 再用 learned rerank 对候选池重排
- 最终得到 top300 query

当前配置核心参数：

- `feat_strides: [4, 8, 16]`
- `proposal_topk_per_level: [180, 90, 30]`
- `proposal_level_score_bias: [0.25, 0.10, 0.0]`
- `proposal_small_box_threshold: 0.025`
- `proposal_small_box_bonus_weight: 0.12`
- `proposal_use_learned_rerank: True`
- `proposal_rank_score_weight: 1.0`

说明：

- 手工 bias 仍保留，但 learned rerank 已接入
- 后续希望 proposal ranking 逐步从规则转向学习式

### 3.4 当前相机分支

当前相机分支仍然是 RouteROI late fusion：

- decoder 不直接融合相机 memory
- 相机是在 decoder 之后介入
- 先 route
- 再做 ROIAlign
- 最后用于分类增强

当前相机分支特点：

- 有显式 camera router
- 有 `none` 类
- 有 `visible_cameras` 多标签监督
- 只对选中的主相机/稀疏相机做 ROI

当前仍然**没有**做的事：

- 相机参与 proposal 主链
- 早期 cross-modal query 更新
- 相机主导 recall

### 3.5 当前 IMU 接入方式

当前已经接入逐帧姿态信息，但只用于投影修正。

姿态来源：

- `gt_filter` 中的 `towers.{tower}.tower_pose_neu.rotation_quat_neu`

当前接入方式：

- quaternion -> rotation matrix / roll-pitch-yaw
- 在雷达 ROI 投到 RGB 前，用 IMU 修正投影

当前只启用：

- `roll`
- `pitch`

当前未启用：

- `yaw`

原因：

- 当前实现是近似投影修正，不是完整动态外参重建
- `yaw` 容易与 BEV 朝向表达发生双重补偿

已观察到的现象：

- 加 IMU 后，RGB ROI 的高度/尺度更合理
- 左右位置仍有残余误差

## 4. 当前训练策略

当前正式训练配置：

- `configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_16e_bs1x8_fp32_test400_onlineval_4km_super4_960.yml`

### 4.1 训练配置

- epoch：`16`
- GPU：`8`
- batch size：`1 / GPU`
- 总 batch size：`8`
- 预训练：`ResNet50_vd_ssld_v2_pretrained.pdparams`
- 学习率：`1e-4`
- warmup：`800 steps`
- 调度：`CosineDecay`
- 在线验证：开启
- 在线验证集：固定 `test400`

### 4.2 当前训练工程策略

关键设置：

- `norm_type: bn`
- `convert_sync_bn: False`
- `find_unused_parameters: True`

说明：

- 之前 8 卡训练在某些版本上出现过中途卡住/死锁问题
- 最终确认 `find_unused_parameters=True` 是当前稳定跑通 8 卡训练的必要条件之一

### 4.3 best 选择策略

当前 best model 规则：

- 主指标：`AP50`
- 打平指标：`mAP50:90`

原因：

- 当前任务极小目标很多
- `AP50` 更贴近“是否打中目标”
- `mAP50:90` 作为更严格的定位质量约束

## 5. 当前结果

### 5.1 历史对比

#### 旧 RouteROI（只改融合，不改 proposal，小目标主链没重做）

完整测试集结果：

- 四分类：`AP50 = 31.12`
- 四分类：`mAP50:90 = 18.82`
- 单分类：`AP50 = 42.73`
- 单分类：`mAP50:90 = 24.25`

#### P2 第一版（未做 16e 收敛修正）

完整测试集结果：

- 四分类：`AP50 = 37.35`
- 四分类：`mAP50:90 = 22.34`
- 单分类：`AP50 = 51.68`
- 单分类：`mAP50:90 = 30.19`

#### 当前 16e 版本（上一轮完整跑通时记录下来的结果）

完整测试集结果：

- 四分类：`AP50 = 39.13`
- 四分类：`mAP50:90 = 23.69`
- 单分类：`AP50 = 53.56`
- 单分类：`mAP50:90 = 31.90`

说明：

- 这一轮完整测试的输出文件后来被误删
- 上述数值是当时完整测试完成后记录下来的最终结果

### 5.2 在线验证 best

上一轮完整 `16e` 训练时，`test400` 在线验证 best 为：

- `AP50 = 39.75`
- `mAP50:90 = 24.80`

说明：

- 相比旧版本，说明当前方案方向有效
- 但整体提升仍然有限，尚未达到“明显优势”

## 6. 当前已知主要问题

### 6.1 当前提升来自哪里

主要来自：

- 小目标雷达主检测链从 `[8,16,32]` 升级到 `[4,8,16]`
- P2 小目标细节分支
- learned proposal rerank
- 更短、更贴合当前结构的训练配方

### 6.2 当前最大瓶颈

当前最大问题已经不再是 route 本身，而是：

- proposal / decoder 的 recall 与定位仍然不够强
- 相机分支还是 late fusion
- 相机不能有效补救 radar 漏检

### 6.3 当前 RGB ROI 问题

当前 RGB 投影分支仍有问题：

- 位置通常接近 GT，但不够准
- 预测框偏小
- IMU 修正后高度好了，但水平误差还在

这说明：

- IMU 修正是有效的
- 但 RGB 投影模型仍然偏粗糙
- 且 GT 质量本身也还有噪声

### 6.4 当前标签问题

目前 RGB 2D GT 仍然存在质量问题：

- 用户已经人工检查并发现部分 GT 不合理
- 新修过的版本目前还没有完整并入正式训练链

这件事会继续限制相机分支上限。

## 7. 当前方案的本质判断

当前方案可以概括为：

**“P2 小目标增强 + learned proposal rerank + IMU roll/pitch 投影修正”的 RouteROI 强化版。**

但是它仍然属于：

**雷达主检测 + 相机 late fusion 分类增强**

并不属于真正“重构融合结构”的版本。

也就是说：

- 当前方案证明：继续强化 radar 主链是有效的
- 但如果想继续明显提升，后续大概率必须开始动融合本身

## 8. 当前运行状态

由于上一轮训练产物被误删，当前正在重新训练同一套 `16e` 配置。

当前重跑日志：

- `output/sealand_radardetr_r50vd_route_roi_p2_16e_bs1x8_fp32_test400_onlineval_4km_super4_960/logs/dist_20260330_100253/workerlog.0`

本次重跑输出目录：

- `output/sealand_radardetr_r50vd_route_roi_p2_16e_bs1x8_fp32_test400_onlineval_4km_super4_960_rerun_20260330`

## 9. 建议外部分析时重点关注的问题

建议重点分析：

1. 当前 late fusion 是否已经到瓶颈
2. 当前 proposal rerank 是否真正学到了小目标质量排序
3. 当前 P2 小目标细节分支是否仍然学得不够充分
4. 当前 RGB ROI 几何误差主要来自：
   - 动态姿态
   - 静态外参残差
   - GT 噪声
   - 或投影模型过粗
5. 下一步是否应该开始重构融合主结构，而不是继续只修 radar 主链

