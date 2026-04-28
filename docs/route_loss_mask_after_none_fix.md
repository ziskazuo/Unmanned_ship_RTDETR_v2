# Route Loss Mask Check After None Fix

## 1. 核心统计

- 数据集: `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- 总目标数: **125033**
- `gt_primary_camera=4` 数量: **0**
- 检查条件: `gt_primary_camera != 4 AND gt_has_camera_box[gt_primary_camera] == 0`
- 命中数量: **171**

| split | count |
| --- | ---: |
| Train | 102 |
| Valid | 31 |
| Test | 38 |

### 分布补充

| gt_primary_camera | count |
| --- | ---: |
| 0(Back) | 24 |
| 1(Front) | 65 |
| 2(Left) | 40 |
| 3(Right) | 42 |

| gt_has_camera_box pattern | count |
| --- | ---: |
| `[0, 0, 0, 0]` | 171 |

- 这 171 个样本的 `label_source` 全部为 `radar_proj.corners_3d`。

## 2. 样本明细

- 全量明细 CSV: `output/route_loss_mask_after_none_fix/primary_with_no_box_details.csv`
- 全量明细 JSONL: `output/route_loss_mask_after_none_fix/primary_with_no_box_details.jsonl`
- 以下展示前 30 条：

| split | sample_id | ann_index | gt_primary_camera | gt_has_camera_box | gt_camera_box_2d_primary |
| --- | --- | ---: | ---: | --- | --- |
| Train | Train/S01/CoastGuard1/000319 | 0 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000367 | 0 | 0 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000372 | 0 | 0 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000379 | 5 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000386 | 2 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000663 | 4 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000674 | 0 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard1/000697 | 0 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000046 | 0 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000199 | 0 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000291 | 0 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000305 | 0 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000325 | 1 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard2/000326 | 1 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000264 | 0 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000294 | 0 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000427 | 3 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000571 | 3 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000585 | 0 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000648 | 6 | 0 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S01/CoastGuard3/000725 | 1 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard1/000650 | 5 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard1/000664 | 3 | 3 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard1/000730 | 2 | 0 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard2/000124 | 1 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard2/000310 | 1 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard2/000441 | 2 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard2/000492 | 2 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/S02/CoastGuard2/000581 | 4 | 1 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |
| Train | Train/T01/CoastGuard1/000426 | 3 | 2 | `[0, 0, 0, 0]` | `[0.0, 0.0, 0.0, 0.0]` |

## 3. Loss Mask 代码核查

核查文件：`ppdet/modeling/losses/detr_loss.py`（`DINOLoss_Rotate_RouteROI._get_route_roi_losses`）

- `primary < 4` 过滤：`valid_primary = primary < 4`（约 L857）
- 按 primary 相机取 `gt_has_camera_box`：`target_has_box = ... > 0.5`（约 L877-L878）
- 仅当 `target_has_box` 为真才进入 camera/proj 监督分支（约 L880-L913）
- `loss_cam_class / loss_proj_l1 / loss_proj_iou` 仅在 `camera_logits_all` 非空时计算（约 L938-L954）

结论：当前 `cam_class / proj_l1 / proj_iou` 的有效监督样本掩码等价于：

`(gt_primary_camera < 4) AND (gt_has_camera_box[gt_primary_camera] == 1)`

因此不会把无效 2D box 样本纳入这三项相机监督损失。

## 4. 是否需要修复

- 本次核查下，loss mask 已符合要求。
- 未对 `DINOLoss_Rotate_RouteROI` 的 mask 逻辑做代码修改。
