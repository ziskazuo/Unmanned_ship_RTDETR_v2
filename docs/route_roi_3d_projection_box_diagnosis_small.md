# RouteROI 3D Projection Box Quick Diagnosis (Small Sample)

- sample_source: `output/route_roi_3d_projection_box_debug_small/samples_600.jsonl` (Train/Valid/Test 各 200)
- num_samples_loaded: **600**
- num_object_rows: **3283**
- num_raw_box_pairs: **3270**

## 快速结论

- `current_projection_box` 在现实现里**没有**执行 `roi_expand_ratio`；这会让框相比预期的 RouteROI 训练逻辑显著偏小。
- `current_projection_box` 底层确实使用了完整 8 点（bottom4 + top4），不是只用单面。
- 小样本未发现 L/W 交换或 yaw 偏 90°：三项疑似计数均为 0。

## 关键指标（小样本）

| metric | mean | median | p10 | p90 |
| --- | ---: | ---: | ---: | ---: |
| current_projection_width_over_raw_width | 0.984760 | 0.990650 | 0.963887 | 0.997820 |
| cuboid_8pt_width_over_raw_width | 1.139900 | 1.146658 | 1.090141 | 1.156673 |
| current_projection_height_over_raw_height | 1.009886 | 1.000233 | 0.960244 | 1.071522 |
| cuboid_8pt_height_over_raw_height | 1.148631 | 1.149054 | 1.129415 | 1.171394 |

补充：对同一小样本复算（仅用于定位原因）显示：
- `current + no_expand`: width≈0.9848, height≈1.0099（接近 raw）
- `current + expand(1.15)`: width≈1.1250, height≈1.1588
- `cuboid_8pt + no_expand`: width≈0.9978, height≈1.0010（与 raw 基本一致）
- `cuboid_8pt + expand(1.15)`: width≈1.1399, height≈1.1486

=> 长度偏短的主因是“缺少 expand”，其次是 current 构造 cuboid 相对 raw corners_3d 还有小幅收缩。

## 产物

- 可视化（50 个大船）: `output/route_roi_3d_projection_box_debug_small/top50_large_ship_visualizations/`
- 可视化清单: `output/route_roi_3d_projection_box_debug_small/top50_visualization_manifest.csv`
- 全量明细: `output/route_roi_3d_projection_box_debug_small/all_raw_box_projection_pairs.csv`
- L/W/yaw 明细: `output/route_roi_3d_projection_box_debug_small/object_level_lw_yaw_diagnosis.csv`
