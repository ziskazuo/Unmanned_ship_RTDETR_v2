# Hard Sector Camera Route Check

- generated_at: `2026-04-24T23:51:46`
- dataset_root: `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- processed_samples: `26400`
- total_objects: `125033`
- skipped_objects: `0`

## Split sample counts

| split | sample_count |
|---|---:|
| Train | 18150 |
| Valid | 4950 |
| Test | 3300 |

## center_angle mode

- overall accuracy: `124490/125033` (99.57%)
- boundary_case accuracy: `10743/11286` (95.19%)
- non_boundary_case accuracy: `113747/113747` (100.00%)

### Per-split accuracy

| split | correct/total | accuracy |
|---|---:|---:|
| Train | 85770/86030 | 99.70% |
| Valid | 23032/23149 | 99.49% |
| Test | 15688/15854 | 98.95% |

### Per-camera accuracy (by GT primary camera)

| gt_camera | correct/total | accuracy |
|---|---:|---:|
| 0 (Back) | 28707/28709 | 99.99% |
| 1 (Front) | 40153/40157 | 99.99% |
| 2 (Left) | 30610/30613 | 99.99% |
| 3 (Right) | 25020/25022 | 99.99% |
| 4 (None) | 0/532 | 0.00% |

### Confusion Matrix (rows=GT, cols=Pred)

| GT \\ Pred | 0(Back) | 1(Front) | 2(Left) | 3(Right) | 4(None) |
|---|---:|---:|---:|---:|---:|
| 0 (Back) | 28707 | 0 | 1 | 1 | 0 |
| 1 (Front) | 0 | 40153 | 2 | 2 | 0 |
| 2 (Left) | 2 | 1 | 30610 | 0 | 0 |
| 3 (Right) | 1 | 1 | 0 | 25020 | 0 |
| 4 (None) | 215 | 224 | 50 | 43 | 0 |

## polygon_area mode

- overall accuracy: `124479/125033` (99.56%)
- boundary_case accuracy: `10732/11286` (95.09%)
- non_boundary_case accuracy: `113747/113747` (100.00%)

### Per-split accuracy

| split | correct/total | accuracy |
|---|---:|---:|
| Train | 85766/86030 | 99.69% |
| Valid | 23032/23149 | 99.49% |
| Test | 15681/15854 | 98.91% |

### Per-camera accuracy (by GT primary camera)

| gt_camera | correct/total | accuracy |
|---|---:|---:|
| 0 (Back) | 28707/28709 | 99.99% |
| 1 (Front) | 40153/40157 | 99.99% |
| 2 (Left) | 30609/30613 | 99.99% |
| 3 (Right) | 25010/25022 | 99.95% |
| 4 (None) | 0/532 | 0.00% |

### Confusion Matrix (rows=GT, cols=Pred)

| GT \\ Pred | 0(Back) | 1(Front) | 2(Left) | 3(Right) | 4(None) |
|---|---:|---:|---:|---:|---:|
| 0 (Back) | 28707 | 0 | 1 | 1 | 0 |
| 1 (Front) | 0 | 40153 | 2 | 2 | 0 |
| 2 (Left) | 2 | 2 | 30609 | 0 | 0 |
| 3 (Right) | 4 | 8 | 0 | 25010 | 0 |
| 4 (None) | 217 | 234 | 41 | 40 | 0 |

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

- center_angle wrong but polygon_area correct: `0`
- polygon_area wrong: `554`
- boundary_case: `11286`

