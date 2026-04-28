# Primary Camera No-Box Fix

- dataset_root: `prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- total_objects: **125033**
- gt_primary_camera=4 count: **0**
- gt_primary_camera != 4 && gt_has_camera_box[primary] == 0 count: **4**

## Fix Stats

- fixed_by_gt_filter_fallback_count: **128**
- fixed_by_3d_projection_count: **0**
- still_no_primary_box_count: **4**
- overwrite_guard_protected_count: **0**

补充（以修复前 171 问题集为基准回溯）：
- from_original_171 fixed_by_gt_filter_fallback_count: **167**
- from_original_171 fixed_by_3d_projection_count: **0**
- from_original_171 still_no_primary_box_count: **4**
- from_original_171 still_no_primary_box_reason: **projection_box_outside_image_or_no_raw = 4**

### still_no_primary_box reason 分布

| reason | count |
| --- | ---: |
| raw_primary_clipped_empty | 4 |

## COCO 同步

| coco_file | changed_fields | missing_sample | mismatch_len |
| --- | ---: | ---: | ---: |
| train_coco.json | 392 | 0 | 0 |
| valid_coco.json | 124 | 0 | 0 |
| test_coco.json | 152 | 0 | 0 |

## Debug 明细

- details csv: `output/primary_camera_no_box_fix_debug/fix_details.csv`
- details jsonl: `output/primary_camera_no_box_fix_debug/fix_details.jsonl`
