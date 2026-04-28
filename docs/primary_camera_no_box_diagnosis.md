# Primary Camera No-Box Diagnosis

- total_issue_count: **171**
- main_stats_source: `gt_filter_only_yaw` (与 None-fix 脚本一致)
- cross_check_source: `gt_filter` (用于判断是否存在可恢复 raw box)

## 1. 四路 raw box 可用性（gt_filter_only_yaw）

- any_camera_has_valid_box: **0 / 171**
- primary_camera_has_valid_box: **0 / 171**
- non_primary_camera_has_valid_box: **0 / 171**

## 2. primary 无 box、其它相机有 box（gt_filter_only_yaw）

- 样本数: **0**

| other_camera_id | count |
| --- | ---: |
| (empty) | 0 |

| boundary_case | count |
| --- | ---: |
| (empty) | 0 |

- hard_sector_center 与 raw_best_camera 不一致: **0**
- hard_sector_center 与 raw_best_camera 一致: **0**

## 3. primary 相机 3D cuboid projection 尝试

- projection_success_count: **0**
- projection_failed_reason 分布：

| reason | count |
| --- | ---: |
| projection_box_outside_image | 4 |
| raw_primary_missing | 167 |

## 4. 失败原因分布（按请求口径）

| reason | count |
| --- | ---: |
| raw_primary_missing | 167 |
| raw_primary_invalid_nan | 0 |
| raw_primary_clipped_empty | 0 |
| raw_primary_too_small | 0 |
| projection_box_outside_image | 4 |
| projection_invalid_depth | 0 |
| unknown | 0 |

## 5. 交叉核查（gt_filter）

- any_camera_has_valid_box(gt_filter): **167 / 171**
- primary_camera_has_valid_box(gt_filter): **167 / 171**
- non_primary_camera_has_valid_box(gt_filter): **0 / 171**
- gt_filter_only_yaw object_missing: **167 / 171**

结论：`gt_filter_only_yaw` 与 `gt_filter` 的 object 可见性存在显著差异，171 个问题样本主要由此触发。

## 6. 是否应 fallback 到 raw_best_camera（建议）

- 当前问题集里，`primary 无 box 但其它相机有 box` **不是多数**。
- 暂不建议把主策略改为 `raw_best_camera` fallback。
- 更推荐修复数据构建/修复脚本的 raw 源选择：在 `gt_filter_only_yaw` 缺对象时，回退到 `gt_filter`（或保留原有非空监督），避免把有效 2D box 覆盖为全 0。

## 7. 明细文件

- summary: `output/primary_camera_no_box_debug/summary.json`
- details csv: `output/primary_camera_no_box_debug/diagnosis_details.csv`
- details jsonl: `output/primary_camera_no_box_debug/diagnosis_details.jsonl`

