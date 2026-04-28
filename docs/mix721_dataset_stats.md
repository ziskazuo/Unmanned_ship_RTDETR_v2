# mix721 dataset stats

## Build setup
- output_root: `/data1/zuokun/code/USV/Unmanned_ship_RTDETR_v2/prepared/sealand_single_tower_2km_super4_1536_route_roi_hbb_min8_mix721`
- split_strategy: `scene/tower + block shuffle + 7:2:1`
- seed: `20260425`
- block_size: `50`
- range_max_m: `2000.0`
- boundary_keep_visible_ratio_threshold: `0.3`
- boundary_keep_min_edge_px_threshold: `8.0`

## Split summary
| split | images | annotations | empty_images | empty_rate |
| --- | --- | --- | --- | --- |
| train | 18150 | 86030 | 14 | 0.08% |
| valid | 4950 | 23149 | 1 | 0.02% |
| test | 3300 | 15854 | 0 | 0.00% |
| total | 26400 | 125033 | 15 | 0.06% |

## Source scene coverage (T/V/S)
| split | from_T | from_V | from_S |
| --- | --- | --- | --- |
| train | 12100 | 1100 | 4950 |
| valid | 3300 | 300 | 1350 |
| test | 2200 | 200 | 900 |

## Class distribution by split
| split | class | count | ratio_in_split |
| --- | --- | --- | --- |
| train | CargoShip | 29803 | 34.64% |
| train | CruiseShip | 14122 | 16.42% |
| train | FishingVessel | 25159 | 29.24% |
| train | RecreationalBoat | 16946 | 19.70% |
| valid | CargoShip | 8156 | 35.23% |
| valid | CruiseShip | 4038 | 17.44% |
| valid | FishingVessel | 6481 | 28.00% |
| valid | RecreationalBoat | 4474 | 19.33% |
| test | CargoShip | 5779 | 36.45% |
| test | CruiseShip | 2534 | 15.98% |
| test | FishingVessel | 4497 | 28.37% |
| test | RecreationalBoat | 3044 | 19.20% |

## Scene/Tower distribution by split (images, annotations)
| split | scene/tower | images | annotations |
| --- | --- | --- | --- |
| train | S01/CoastGuard1 | 550 | 3138 |
| train | S01/CoastGuard2 | 550 | 2151 |
| train | S01/CoastGuard3 | 550 | 2848 |
| train | S02/CoastGuard1 | 550 | 3506 |
| train | S02/CoastGuard2 | 550 | 3774 |
| train | S03/CoastGuard1 | 550 | 1275 |
| train | S03/CoastGuard2 | 550 | 1074 |
| train | S03/CoastGuard3 | 550 | 908 |
| train | S03/CoastGuard4 | 550 | 1354 |
| train | T01/CoastGuard1 | 550 | 4936 |
| train | T01/CoastGuard2 | 550 | 5199 |
| train | T01/CoastGuard3 | 550 | 5679 |
| train | T01/CoastGuard4 | 550 | 4239 |
| train | T02/CoastGuard1 | 550 | 2490 |
| train | T02/CoastGuard2 | 550 | 2101 |
| train | T02/CoastGuard3 | 550 | 2168 |
| train | T03/CoastGuard1 | 550 | 3003 |
| train | T03/CoastGuard2 | 550 | 3441 |
| train | T04/CoastGuard1 | 550 | 1635 |
| train | T04/CoastGuard2 | 550 | 1320 |
| train | T04/CoastGuard3 | 550 | 1665 |
| train | T05/CoastGuard1 | 550 | 2521 |
| train | T05/CoastGuard2 | 550 | 2811 |
| train | T06/CoastGuard1 | 550 | 1972 |
| train | T06/CoastGuard2 | 550 | 2359 |
| train | T06/CoastGuard3 | 550 | 1858 |
| train | T07/CoastGuard1 | 550 | 2317 |
| train | T07/CoastGuard2 | 550 | 2215 |
| train | T08/CoastGuard1 | 550 | 1284 |
| train | T08/CoastGuard2 | 550 | 2144 |
| train | T08/CoastGuard3 | 550 | 2276 |
| train | V01/CoastGuard1 | 550 | 3110 |
| train | V01/CoastGuard2 | 550 | 3259 |
| valid | S01/CoastGuard1 | 150 | 894 |
| valid | S01/CoastGuard2 | 150 | 735 |
| valid | S01/CoastGuard3 | 150 | 820 |
| valid | S02/CoastGuard1 | 150 | 1134 |
| valid | S02/CoastGuard2 | 150 | 1182 |
| valid | S03/CoastGuard1 | 150 | 423 |
| valid | S03/CoastGuard2 | 150 | 151 |
| valid | S03/CoastGuard3 | 150 | 375 |
| valid | S03/CoastGuard4 | 150 | 297 |
| valid | T01/CoastGuard1 | 150 | 1074 |
| valid | T01/CoastGuard2 | 150 | 1434 |
| valid | T01/CoastGuard3 | 150 | 1193 |
| valid | T01/CoastGuard4 | 150 | 971 |
| valid | T02/CoastGuard1 | 150 | 703 |
| valid | T02/CoastGuard2 | 150 | 465 |
| valid | T02/CoastGuard3 | 150 | 409 |
| valid | T03/CoastGuard1 | 150 | 900 |
| valid | T03/CoastGuard2 | 150 | 1037 |
| valid | T04/CoastGuard1 | 150 | 440 |
| valid | T04/CoastGuard2 | 150 | 355 |
| valid | T04/CoastGuard3 | 150 | 506 |
| valid | T05/CoastGuard1 | 150 | 531 |
| valid | T05/CoastGuard2 | 150 | 462 |
| valid | T06/CoastGuard1 | 150 | 534 |
| valid | T06/CoastGuard2 | 150 | 786 |
| valid | T06/CoastGuard3 | 150 | 536 |
| valid | T07/CoastGuard1 | 150 | 585 |
| valid | T07/CoastGuard2 | 150 | 587 |
| valid | T08/CoastGuard1 | 150 | 487 |
| valid | T08/CoastGuard2 | 150 | 640 |
| valid | T08/CoastGuard3 | 150 | 800 |
| valid | V01/CoastGuard1 | 150 | 845 |
| valid | V01/CoastGuard2 | 150 | 858 |
| test | S01/CoastGuard1 | 100 | 802 |
| test | S01/CoastGuard2 | 100 | 488 |
| test | S01/CoastGuard3 | 100 | 474 |
| test | S02/CoastGuard1 | 100 | 807 |
| test | S02/CoastGuard2 | 100 | 730 |
| test | S03/CoastGuard1 | 100 | 228 |
| test | S03/CoastGuard2 | 100 | 185 |
| test | S03/CoastGuard3 | 100 | 200 |
| test | S03/CoastGuard4 | 100 | 190 |
| test | T01/CoastGuard1 | 100 | 563 |
| test | T01/CoastGuard2 | 100 | 903 |
| test | T01/CoastGuard3 | 100 | 1159 |
| test | T01/CoastGuard4 | 100 | 784 |
| test | T02/CoastGuard1 | 100 | 520 |
| test | T02/CoastGuard2 | 100 | 256 |
| test | T02/CoastGuard3 | 100 | 414 |
| test | T03/CoastGuard1 | 100 | 670 |
| test | T03/CoastGuard2 | 100 | 524 |
| test | T04/CoastGuard1 | 100 | 295 |
| test | T04/CoastGuard2 | 100 | 195 |
| test | T04/CoastGuard3 | 100 | 298 |
| test | T05/CoastGuard1 | 100 | 421 |
| test | T05/CoastGuard2 | 100 | 453 |
| test | T06/CoastGuard1 | 100 | 485 |
| test | T06/CoastGuard2 | 100 | 447 |
| test | T06/CoastGuard3 | 100 | 259 |
| test | T07/CoastGuard1 | 100 | 486 |
| test | T07/CoastGuard2 | 100 | 465 |
| test | T08/CoastGuard1 | 100 | 225 |
| test | T08/CoastGuard2 | 100 | 604 |
| test | T08/CoastGuard3 | 100 | 276 |
| test | V01/CoastGuard1 | 100 | 481 |
| test | V01/CoastGuard2 | 100 | 567 |

## Scene/Tower split ratio (train/valid/test)
| scene/tower | train_images | valid_images | test_images | train_ratio | valid_ratio | test_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| S01/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S01/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S01/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S02/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S02/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S03/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S03/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S03/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| S03/CoastGuard4 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T01/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T01/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T01/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T01/CoastGuard4 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T02/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T02/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T02/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T03/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T03/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T04/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T04/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T04/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T05/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T05/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T06/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T06/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T06/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T07/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T07/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T08/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T08/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| T08/CoastGuard3 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| V01/CoastGuard1 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |
| V01/CoastGuard2 | 550 | 150 | 100 | 68.75% | 18.75% | 12.50% |

## RouteROI coverage
| field | valid_annotations | coverage |
| --- | --- | --- |
| gt_primary_camera | 125033 | 100.00% |
| gt_visible_cameras | 125033 | 100.00% |
| gt_camera_box_2d | 125033 | 100.00% |
| gt_has_camera_box | 125033 | 100.00% |

## gt_primary_camera distribution
| id | camera | count | ratio |
| --- | --- | --- | --- |
| 0 | Back | 28709 | 22.96% |
| 1 | Front | 40157 | 32.12% |
| 2 | Left | 30613 | 24.48% |
| 3 | Right | 25022 | 20.01% |
| 4 | None | 532 | 0.43% |

## gt_visible_cameras count distribution
| num_visible_cameras | count | ratio |
| --- | --- | --- |
| 0 | 532 | 0.43% |
| 1 | 124501 | 99.57% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |
| 4 | 0 | 0.00% |

## Radar boundary clipping stats
| metric | value |
| --- | --- |
| total_objects | 136689 |
| center_in_2km_objects | 125062 |
| full_inside_radar_objects | 124665 |
| boundary_crossing_objects | 397 |
| clipped_keep_objects | 368 |
| clipped_drop_visible_ratio_objects | 0 |
| clipped_drop_min8_objects | 29 |

| visible_ratio_stat | value |
| --- | --- |
| mean | 0.823691 |
| median | 0.832255 |
| p10 | 0.650881 |
| p25 | 0.742975 |

## All filtering reasons
| filter_reason | count |
| --- | --- |
| drop_center_out_2km | 11627 |
| drop_clipped_min_edge_lt_threshold | 29 |

## Block leakage check (adjacent frames)
| scene/tower | adjacent_pairs | adjacent_split_changes | adjacent_cross_block_changes | change_rate |
| --- | --- | --- | --- | --- |
| S01/CoastGuard1 | 799 | 7 | 7 | 0.88% |
| S01/CoastGuard2 | 799 | 8 | 8 | 1.00% |
| S01/CoastGuard3 | 799 | 8 | 8 | 1.00% |
| S02/CoastGuard1 | 799 | 7 | 7 | 0.88% |
| S02/CoastGuard2 | 799 | 6 | 6 | 0.75% |
| S03/CoastGuard1 | 799 | 5 | 5 | 0.63% |
| S03/CoastGuard2 | 799 | 8 | 8 | 1.00% |
| S03/CoastGuard3 | 799 | 8 | 8 | 1.00% |
| S03/CoastGuard4 | 799 | 9 | 9 | 1.13% |
| T01/CoastGuard1 | 799 | 8 | 8 | 1.00% |
| T01/CoastGuard2 | 799 | 7 | 7 | 0.88% |
| T01/CoastGuard3 | 799 | 6 | 6 | 0.75% |
| T01/CoastGuard4 | 799 | 10 | 10 | 1.25% |
| T02/CoastGuard1 | 799 | 7 | 7 | 0.88% |
| T02/CoastGuard2 | 799 | 6 | 6 | 0.75% |
| T02/CoastGuard3 | 799 | 7 | 7 | 0.88% |
| T03/CoastGuard1 | 799 | 8 | 8 | 1.00% |
| T03/CoastGuard2 | 799 | 7 | 7 | 0.88% |
| T04/CoastGuard1 | 799 | 6 | 6 | 0.75% |
| T04/CoastGuard2 | 799 | 8 | 8 | 1.00% |
| T04/CoastGuard3 | 799 | 10 | 10 | 1.25% |
| T05/CoastGuard1 | 799 | 3 | 3 | 0.38% |
| T05/CoastGuard2 | 799 | 7 | 7 | 0.88% |
| T06/CoastGuard1 | 799 | 8 | 8 | 1.00% |
| T06/CoastGuard2 | 799 | 9 | 9 | 1.13% |
| T06/CoastGuard3 | 799 | 9 | 9 | 1.13% |
| T07/CoastGuard1 | 799 | 9 | 9 | 1.13% |
| T07/CoastGuard2 | 799 | 8 | 8 | 1.00% |
| T08/CoastGuard1 | 799 | 8 | 8 | 1.00% |
| T08/CoastGuard2 | 799 | 9 | 9 | 1.13% |
| T08/CoastGuard3 | 799 | 10 | 10 | 1.25% |
| V01/CoastGuard1 | 799 | 7 | 7 | 0.88% |
| V01/CoastGuard2 | 799 | 9 | 9 | 1.13% |

- overall_adjacent_pairs: `26367`
- overall_adjacent_split_changes: `252` (0.96%)
- overall_adjacent_cross_block_changes: `252` (0.96%)

## Acceptance checks
- train/valid/test all cover original T/V/S scenes: `True`
- scene/tower ratio near 7:2:1 (tolerance train[0.60,0.80], valid[0.10,0.30], test[0.05,0.20]): `33/33`
- no frame-level random leakage (adjacent split-change mostly at block boundaries): `overall change rate 0.96%`
- 2km center filter still enforced: `True`
- boundary crossing annotations clipped and keep/drop handled: `True`
- RouteROI supervision fields complete: `True`
