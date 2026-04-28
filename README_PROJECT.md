# Radar + Multi-Camera RT-DETR Project (Pre-GitHub Publish)

## Project Overview

本项目基于 RT-DETR 做雷达与多相机融合检测，目标是在 **RGB 图像坐标系** 输出 2D 检测框，并以 COCO 风格 AP 指标评估。

- Task: radar-assisted multi-camera object detection
- Input:
  - 多视角 RGB 图像
  - 雷达检测信息（作为先验/辅助）
- Output:
  - RGB 图像空间下的 2D bbox
- Evaluation:
  - COCO-style AP（含 AP50 / AP75 / mAP50:90）

说明：雷达信息用于辅助提升检测（特别是小目标），最终输出框以 RGB 图像空间为准。

---

## Best Checkpoint

当前用于发布说明的最优指标如下：

| Metric | Value |
|---|---:|
| AP50 | 78.51% |
| AP75 | 49.13% |
| mAP50:90 | 54.89% |

对应日志位置：

- `output/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/logs/train_gpus_0_1_2_3_20260426_215729.log`（`[04/27 15:05:45]`）

当前权重文件（实际路径）：

- `output/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/best_model.pdparams`
- `output/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/best_model.pdema`
- `output/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/best_model.pdopt`

sha256（建议发布时附带）：

- `best_model.pdparams`: `9cd8e54a65fa0c1e1d23f4d088eb454ce7873b7471e0d1f4f3d383eeadba99c0`
- `best_model.pdema`: `1129b298162a70ceb73115820f0c435f855b5a3b0c8375a24395b39ded8dc05e`
- `best_model.pdopt`: `76b14e808bc608c6270fa527c12f734a4f6d345bfc086489d8ff90163bdbfff7`

---

## Repository Structure

> 仅保留与训练/评估/复现相关的主干结构（省略缓存、临时日志与大量中间输出）。

```text
.
├── configs/
│   ├── datasets/
│   │   ├── sealand_single_tower_radarcamera_2km_super4_1536_route_roi_hbb_min8_mix721.yml
│   │   └── sealand_single_tower_*.yml
│   └── rtdetr/
│       ├── sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721.yml
│       └── _base_/
├── ppdet/
│   ├── modeling/
│   ├── data/
│   ├── engine/
│   └── metrics/
├── tools/
│   ├── train.py
│   ├── eval.py
│   ├── eval_rbox_coco_style.py
│   ├── build_sealand_single_tower_dataset.py
│   ├── build_sealand_single_tower_dataset_mix721.py
│   └── wait_and_launch_sealand_multigpu.py
├── prepared/
│   └── sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721/
├── output/
│   ├── sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721/
│   ├── eval_*/
│   └── gpu_queue/
├── docs/
│   ├── none_primary_camera_diagnosis_and_fix.md
│   ├── primary_camera_no_box_fix.md
│   └── route_roi_3d_projection_box_diagnosis_small.md
└── README_PROJECT.md
```

---

## Key Files And Roles

### 1) 模型结构相关（Architecture / Transformer / Head / Loss）

- `ppdet/modeling/architectures/radarcamera_detr.py`
  - `RadarCamera_RouteROI_DETR` 主体架构（雷达+相机融合检测框架）
- `ppdet/modeling/transformers/rtdetr_transformer.py`
  - `RTDETRTransformer_Rotate_RouteROI`，融合与解码核心
- `ppdet/modeling/heads/detr_head.py`
  - `DINOHead_Rotate_RouteROI`，输出分类/框与路线相关监督头
- `ppdet/modeling/losses/detr_loss.py`
  - `DINOLoss_Rotate_RouteROI`，训练损失定义
- `ppdet/modeling/transformers/matchers.py`
  - `HungarianMatcher_Rotate`，匹配器
- `ppdet/modeling/post_process.py`
  - 后处理（推理阶段结果整理）

### 2) 数据读取、增强与 GT 构建

- `ppdet/data/source/coco.py`
  - `CameraRadar_COCODataSet` 数据集读取与字段组织
- `ppdet/data/reader.py`
  - Train/Eval/Test Reader 构建
- `ppdet/data/transform/operators.py`
  - 常用图像/标注变换
- `ppdet/data/transform/rotated_operators.py`
  - 旋转框相关变换
- `tools/build_sealand_single_tower_dataset.py`
  - 单塔数据集构建（基础版）
- `tools/build_sealand_single_tower_dataset_mix721.py`
  - mix721 数据集构建流程
- `tools/yaml_to_coco_allclass.py`
  - 标注转换为 COCO 格式（全类别）

### 3) 训练与评估入口

- `tools/train.py`
  - 训练入口
- `tools/eval.py`
  - 通用评估入口
- `tools/eval_rbox_coco_style.py`
  - RBox COCO 风格指标评估
- `tools/eval_ship_only.py`
  - ship-only 指标评估
- `tools/train_sealand_gpu4.sh`
  - 多卡训练封装脚本
- `tools/wait_and_launch_sealand_multigpu.py`
  - GPU 队列等待与自动拉起

### 4) 配置文件（建议优先关注）

- `configs/rtdetr/sealand_radardetr_r50vd_route_roi_p2_2km_super4_r1536_c1024x512_hbb_min8_mix721.yml`
  - 当前核心实验配置（epoch、metric、save_dir、tie-break 等）
- `configs/rtdetr/_base_/radardetr_r50vd_route_roi_p2.yml`
  - 模型组件装配与默认超参
- `configs/datasets/sealand_single_tower_radarcamera_2km_super4_1536_route_roi_hbb_min8_mix721.yml`
  - 数据根目录与 Train/Eval/Test 标注文件

### 5) 诊断 / 可视化 / 调试脚本

- `tools/diagnose_primary_camera_no_box.py`
- `tools/fix_primary_camera_no_box_mix721.py`
- `tools/diagnose_and_fix_none_primary_camera_mix721.py`
- `tools/diagnose_route_roi_3d_projection_box.py`
- `tools/check_route_roi_3d_projection_quality.py`
- `tools/check_hard_sector_camera_route.py`
- `tools/visualize_eval.py`
- `tools/visualize_frame_multiview.py`

### 6) 关键文档

- `docs/none_primary_camera_diagnosis_and_fix.md`
- `docs/primary_camera_no_box_fix.md`
- `docs/route_roi_3d_projection_box_diagnosis_small.md`
- `docs/mix721_dataset_stats.md`

---

## Best Weight Release Plan (GitHub)

考虑到权重文件体积较大（单文件 >100MB），建议不要直接走普通 Git 上传，而是采用以下方案之一：

1. **GitHub Release Assets（推荐）**
   - 在代码仓库版本标签（tag）下上传 `best_model.pdparams/.pdema/.pdopt`
   - 在 README 中保留下载链接与 sha256

2. **Git LFS**
   - 将 `*.pdparams`, `*.pdema`, `*.pdopt` 纳入 LFS 管理
   - 适合需要在仓库里直接按路径引用权重的场景

建议发布目录约定（逻辑路径，可在发布时执行复制或软链）：

- `weights/best_model_AP50_78.51/best_model.pdparams`
- `weights/best_model_AP50_78.51/best_model.pdema`
- `weights/best_model_AP50_78.51/best_model.pdopt`

---

## Scope Notes

本整理仅做 **上传前结构梳理与文档说明**：

- 未重构模型代码
- 未修改训练/评估逻辑
- 未改动已有实验结果
- 未删除历史输出

