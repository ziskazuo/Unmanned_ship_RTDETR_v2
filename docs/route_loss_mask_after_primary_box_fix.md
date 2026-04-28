# Route Loss Mask Check After Primary Box Fix

- 数据集: `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- 总目标数: **125033**
- `gt_primary_camera=4` 数量: **0**
- 检查条件: `gt_primary_camera != 4 AND gt_has_camera_box[gt_primary_camera] == 0`
- 命中数量: **4**

| split | count |
| --- | ---: |
| Train | 4 |
| Valid | 0 |
| Test | 0 |

## Loss Mask 代码核查

核查文件：`ppdet/modeling/losses/detr_loss.py`（`DINOLoss_Rotate_RouteROI._get_route_roi_losses`）

- `valid_primary = primary < 4`（约 L857）
- `target_has_box = gather(gt_has_camera_box, primary) > 0.5`（约 L877-L878）
- 仅 `target_has_box` 为真才进入 `cam_class/proj_l1/proj_iou`（约 L880-L954）

结论：`cam_class/proj_l1/proj_iou` mask 仍是：

`(gt_primary_camera < 4) AND (gt_has_camera_box[gt_primary_camera] == 1)`

## 明细

- summary: `output/route_loss_mask_after_primary_box_fix/summary.json`
- details csv: `output/route_loss_mask_after_primary_box_fix/primary_with_no_box_details.csv`
- details jsonl: `output/route_loss_mask_after_primary_box_fix/primary_with_no_box_details.jsonl`
