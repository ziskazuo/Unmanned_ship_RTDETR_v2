# RouteROI 3D Projection Box Diagnosis

- dataset_root: `prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- num_samples_loaded: **26400**
- num_raw_box_pairs: **124862**
- num_object_rows: **125033**
- num_topk_visualizations: **50**
- projection_plane_height: **-6.0000**
- roi_expand_ratio: **1.1500**

## Task 1: Fixed Visualization Box Source Audit

| label | source |
| --- | --- |
| raw_box | raw `cams[*].box_2d` from `gt_filter_only_yaw` |
| current_projection_box | current fix-chain cuboid build (`center+extent+yaw+height`) then projection (no roi expand) |
| cuboid_8pt_box | projected from raw `radar_proj.corners_3d` (8 points) + `roi_expand_ratio` |
| gt_camera_box_2d | prepared annotation `gt_camera_box_2d[camera_id]` |

说明：现有 `output/none_primary_camera_fix_debug/fixed/` 历史图仅绘制了 `raw_box` 与 `current_projection_box`。
本次新增 top50 诊断图在图例、文件名、manifest 中显式标注了四类框。

## Task 2: 8-Corner Usage Check

- objects_checked: **125033**
- current_projection_success: **124861**
- current_projection_points_total_is_8: **125033**
- current_projection_points_total_not_8: **0**
- 代码核查：`current_projection_box` 由 bottom 4 点 + top 4 点共 8 点参与 min/max 包围盒计算。

## Task 3: Top 50 Large-Ship Visualization

- visualization_dir: `output/route_roi_3d_projection_box_debug/top50_large_ship_visualizations`
- visualization_manifest: `output/route_roi_3d_projection_box_debug/top50_visualization_manifest.csv`
- 每张图左侧为 RGB 叠框，右侧为雷达 BEV + 对应 rbox。

## Task 4: Width/Height Ratio Statistics

| metric | valid_count | mean | median | p10 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_projection_width_over_raw_width | 124861 | 0.993279 | 0.997561 | 0.981782 | 0.999629 |
| cuboid_8pt_width_over_raw_width | 124862 | 1.142684 | 1.149703 | 1.117180 | 1.153397 |
| current_projection_height_over_raw_height | 124861 | 0.996174 | 0.999962 | 0.972399 | 1.011039 |
| cuboid_8pt_height_over_raw_height | 124862 | 1.149306 | 1.149993 | 1.140933 | 1.157193 |

分桶统计文件：
- `output/route_roi_3d_projection_box_debug/stats_by_category.csv`
- `output/route_roi_3d_projection_box_debug/stats_by_distance_bin.csv`
- `output/route_roi_3d_projection_box_debug/stats_by_camera.csv`
- `output/route_roi_3d_projection_box_debug/stats_by_long_edge_bin.csv`
- `output/route_roi_3d_projection_box_debug/stats_by_all_buckets.csv`

## Task 5: L/W/Yaw Diagnosis

- lw_swap_suspected_count: **0**
- yaw_off90_suspected_count: **0**
- extent_vs_bbox_swap_suspected_count: **0**
- object_level_csv: `output/route_roi_3d_projection_box_debug/object_level_lw_yaw_diagnosis.csv`

## Artifacts

- summary_json: `output/route_roi_3d_projection_box_debug/summary.json`
- all_pairs_csv: `output/route_roi_3d_projection_box_debug/all_raw_box_projection_pairs.csv`
- report_generated_by: `tools/diagnose_route_roi_3d_projection_box.py`

