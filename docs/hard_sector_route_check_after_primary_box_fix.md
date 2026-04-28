# Hard Sector Camera Route Check

- generated_at: `2026-04-25T07:23:42`
- dataset_root: `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- processed_samples: `26400`
- total_objects: `125033`
- skipped_objects: `0`
- gt_primary_camera=4 count: `0`

## Split sample counts

| split | sample_count |
|---|---:|
| Train | 18150 |
| Valid | 4950 |
| Test | 3300 |

## center_angle mode

- overall accuracy: `125021/125033` (99.99%)
- boundary_case accuracy: `11274/11286` (99.89%)
- non_boundary_case accuracy: `113747/113747` (100.00%)

### Per-split accuracy

| split | correct/total | accuracy |
|---|---:|---:|
| Train | 86024/86030 | 99.99% |
| Valid | 23147/23149 | 99.99% |
| Test | 15850/15854 | 99.97% |

### Per-camera accuracy (by GT primary camera)

| gt_camera | correct/total | accuracy |
|---|---:|---:|
| 0 (Back) | 28922/28924 | 99.99% |
| 1 (Front) | 40377/40382 | 99.99% |
| 2 (Left) | 30660/30663 | 99.99% |
| 3 (Right) | 25062/25064 | 99.99% |
| 4 (None) | 0/0 | n/a |

### Confusion Matrix (rows=GT, cols=Pred)

| GT \\ Pred | 0(Back) | 1(Front) | 2(Left) | 3(Right) | 4(None) |
|---|---:|---:|---:|---:|---:|
| 0 (Back) | 28922 | 0 | 1 | 1 | 0 |
| 1 (Front) | 0 | 40377 | 2 | 3 | 0 |
| 2 (Left) | 2 | 1 | 30660 | 0 | 0 |
| 3 (Right) | 1 | 1 | 0 | 25062 | 0 |
| 4 (None) | 0 | 0 | 0 | 0 | 0 |

## polygon_area mode

- overall accuracy: `124998/125033` (99.97%)
- boundary_case accuracy: `11251/11286` (99.69%)
- non_boundary_case accuracy: `113747/113747` (100.00%)

### Per-split accuracy

| split | correct/total | accuracy |
|---|---:|---:|
| Train | 86009/86030 | 99.98% |
| Valid | 23147/23149 | 99.99% |
| Test | 15842/15854 | 99.92% |

### Per-camera accuracy (by GT primary camera)

| gt_camera | correct/total | accuracy |
|---|---:|---:|
| 0 (Back) | 28921/28924 | 99.99% |
| 1 (Front) | 40378/40382 | 99.99% |
| 2 (Left) | 30650/30663 | 99.96% |
| 3 (Right) | 25049/25064 | 99.94% |
| 4 (None) | 0/0 | n/a |

### Confusion Matrix (rows=GT, cols=Pred)

| GT \\ Pred | 0(Back) | 1(Front) | 2(Left) | 3(Right) | 4(None) |
|---|---:|---:|---:|---:|---:|
| 0 (Back) | 28921 | 0 | 1 | 2 | 0 |
| 1 (Front) | 0 | 40378 | 2 | 2 | 0 |
| 2 (Left) | 5 | 8 | 30650 | 0 | 0 |
| 3 (Right) | 4 | 11 | 0 | 25049 | 0 |
| 4 (None) | 0 | 0 | 0 | 0 | 0 |

## sector_main_ratio distribution

- count: `125033`
- quantiles: min=0.3423, p5=0.8185, p10=1.0000, p25=1.0000, p50=1.0000, p75=1.0000, p90=1.0000, p95=1.0000, max=1.0000

| bucket | count |
|---|---:|
| [0.0,0.2) | 0 |
| [0.2,0.4) | 6 |
| [0.4,0.6) | 2052 |
| [0.6,0.8) | 3850 |
| [0.8,0.9) | 1990 |
| [0.9,1.0] | 117135 |

## Focus debug sample counts

- center_angle wrong but polygon_area correct: `1`
- polygon_area wrong: `35`
- boundary_case: `11286`

