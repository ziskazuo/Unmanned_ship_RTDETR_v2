# Z Axis Direction Check

## Scope
- Data source: raw `radar_pcd`, `gt_filter`, `gt_filter_only_yaw`, `gt_sensor`, `opv2v_yaml`.
- Output directory: `output/z_axis_direction_debug`.
- Projection validation samples: 12.

## 1) Raw radar PCD z statistics
- Parsed `FIELDS` from each sampled PCD and extracted `x/y/z`.
- Per-sample stats and in-box stats saved to `pcd_z_stats.csv`.

| sample_id | num_points | z_min | z_p1 | z_median | z_p99 | z_max | inside_object | inside_points | inside_z_median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Train/T01/CoastGuard2/000687 | 6016 | -9.702164 | -6.262891 | 33.509832 | 179.672402 | 211.596433 | USV_Containership01_C_12 | 1026 | 4.996298 |
| Test/S03/CoastGuard2/000344 | 6032 | -6.378084 | -5.512422 | 56.616991 | 191.267497 | 208.276794 | USV_queenmarry_C_8 | 140 | 5.511240 |
| Train/T03/CoastGuard1/000690 | 6512 | -13.357834 | -9.071052 | 20.192291 | 172.643319 | 204.726581 | USV_Containership01_C_20 | 158 | 0.000000 |
| Test/S03/CoastGuard2/000364 | 6175 | -6.294826 | -4.856400 | 60.288114 | 193.892026 | 212.119904 | USV_queenmarry_C_8 | 132 | 5.448699 |
| Train/T07/CoastGuard1/000390 | 1580 | -15.427776 | -0.745781 | 62.337619 | 214.378226 | 248.310356 | USV_libertyship_C_0 | 15 | 10.170447 |
| Valid/V01/CoastGuard1/000249 | 4920 | -69.300874 | -52.494252 | 9.972214 | 282.171908 | 354.078626 | USV_Containership01_C_4 | 1196 | -10.423363 |
| Train/T01/CoastGuard2/000788 | 9854 | -9.804289 | -7.015532 | 11.250931 | 173.296238 | 199.594545 | USV_Containership01_C_12 | 301 | 6.299157 |
| Train/T04/CoastGuard3/000438 | 3539 | -102.159955 | -56.117571 | 10.729987 | 256.857092 | 297.298177 | USV_queenmarry_C_0 | 278 | 33.940174 |
| Train/T08/CoastGuard2/000311 | 6808 | -10.061414 | -7.221779 | 41.769943 | 177.285122 | 208.300905 | USV_Containership02_C_0 | 154 | 3.083172 |
| Test/S01/CoastGuard2/000705 | 4476 | -69.297860 | -27.763864 | 27.313246 | 246.229884 | 311.681490 | USV_queenmarry_C_0 | 46 | -5.766975 |
| Test/S02/CoastGuard2/000112 | 2528 | -49.200546 | -42.948537 | 41.458394 | 197.260410 | 218.741907 | USV_Containership01_C_4 | 158 | 13.576542 |
| Train/T07/CoastGuard2/000087 | 611 | -34.693126 | -33.711267 | 19.210403 | 180.150573 | 182.371522 | USV_queenmarry_C_4 | 31 | 36.618173 |

## 2) Raw 3D box annotation field check
- Field snapshots from one sample file for each source:
- `gt_filter_path`: path=`/data1/liziao/USV/dataset/sealand_data/dataset/Train/T01/gt_filter/gt_000000_tosensor_filter.json`
  top_keys=['tick', 'towers']
  tower_keys=['tower_pose_neu', 'radar_extrinsics_neu', 'objects']
  sample_object=USV_Containership01_C_12
  object_keys=['world_pose_neu', 'rotation_quat_neu', 'bbox_m', 'bev_rot_only_yaw', 'radar_proj', 'cams']
  sample_values={'bbox_m': {'L': 251.37, 'W': 35.44, 'H': 46.55}, 'radar_proj.center': {'x': -310.79553946036003, 'y': 269.4919223972535, 'z': 5.274999999999999}, 'radar_proj.extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}, 'radar_proj.corners_3d[0]': [-293.70881045738093, 143.71926987578377, 28.549999999999997], 'bev_rot_only_yaw': {'center': {'x': -310.7955504882708, 'y': 269.49193195959435, 'z': 11.274999999999999}, 'yaw': -1.5758331113715032, 'extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}}}
- `gt_filter_only_yaw_path`: path=`/data1/liziao/USV/dataset/sealand_data/dataset/Train/T01/gt_filter_only_yaw/gt_000000_tosensor_filter.json`
  top_keys=['tick', 'towers']
  tower_keys=['tower_pose_neu', 'radar_extrinsics_neu', 'objects']
  sample_object=USV_Containership01_C_12
  object_keys=['world_pose_neu', 'rotation_quat_neu', 'bbox_m', 'bev_rot_only_yaw', 'radar_proj', 'cams']
  sample_values={'bbox_m': {'L': 251.37, 'W': 35.44, 'H': 46.55}, 'radar_proj.center': {'x': -310.79553946036003, 'y': 269.4919223972535, 'z': 5.274999999999999}, 'radar_proj.extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}, 'radar_proj.corners_3d[0]': [-293.70881043013213, 143.71926967521168, 28.549999999999997], 'bev_rot_only_yaw': {'center': {'x': -310.7955504882708, 'y': 269.49193195959435, 'z': 11.274999999999999}, 'yaw': -1.5758331113715032, 'extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}}}
- `gt_sensor_path`: path=`/data1/liziao/USV/dataset/sealand_data/dataset/Train/T01/gt_sensor/gt_000000_tosensor.json`
  top_keys=['tick', 'towers']
  tower_keys=['tower_pose_neu', 'radar_extrinsics_neu', 'objects']
  sample_object=USV_Containership01_C_12
  object_keys=['world_pose_neu', 'rotation_quat_neu', 'bbox_m', 'bev_rot_only_yaw', 'radar_proj', 'cams']
  sample_values={'bbox_m': {'L': 251.37, 'W': 35.44, 'H': 46.55}, 'radar_proj.center': {'x': -310.79553946036003, 'y': 269.4919223972535, 'z': 5.274999999999999}, 'radar_proj.extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}, 'radar_proj.corners_3d[0]': [-293.70881045738093, 143.71926987578377, 28.549999999999997], 'bev_rot_only_yaw': {'center': {'x': -310.7955504882708, 'y': 269.49193195959435, 'z': 11.274999999999999}, 'yaw': -1.5758331113715032, 'extent': {'x': 125.685, 'y': 17.72, 'z': 23.275}}}

- Per-target z semantic rows saved to `box_z_semantics.csv` (includes `center_z`, `height`, `possible_bottom_z=center_z-height/2`, `possible_top_z=center_z+height/2`).
- Median |center_z - (z_min+z_max)/2| = 0.000000
- Median min(|center_z-z_min|, |center_z-z_max|) = 14.805000
- Median |height - (z_max-z_min)| = 0.128446
- Inference: radar_proj.center.z behaves as **center_z**.

## 3) Projection A/B test on primary camera
- A hypothesis: `top_z = bottom_z + height` (blue).
- B hypothesis: `top_z = bottom_z - height` (red).
- Visualizations are under `vis/` and include GT 2D box (green), A (blue), B (red), and mean-v text.

| # | sample_id | object | camera | mean_v_bottom_A | mean_v_top_A | A(top<bottom) | mean_v_bottom_B | mean_v_top_B | B(top<bottom) | vis |
|---:|---|---|---|---:|---:|---|---:|---:|---|---|
| 1 | Train/T03/CoastGuard1/000047 | USV_Sailboat01_C_0 | CamFront | 273.908 | 234.141 | True | 234.141 | 273.908 | False | `output/z_axis_direction_debug/vis/Train__T03__CoastGuard1__000047__USV_Sailboat01_C_0__CamFront.png` |
| 2 | Valid/V01/CoastGuard1/000511 | USV_queenmarry_C_0 | CamRight | 280.886 | 249.258 | True | 249.258 | 280.886 | False | `output/z_axis_direction_debug/vis/Valid__V01__CoastGuard1__000511__USV_queenmarry_C_0__CamRight.png` |
| 3 | Train/T01/CoastGuard3/000119 | USV_libertyship_C_0 | CamRight | 267.701 | 229.740 | True | 229.740 | 267.701 | False | `output/z_axis_direction_debug/vis/Train__T01__CoastGuard3__000119__USV_libertyship_C_0__CamRight.png` |
| 4 | Test/S03/CoastGuard3/000174 | USV_Containership01_C_8 | CamLeft | 307.101 | 179.897 | True | 179.897 | 307.101 | False | `output/z_axis_direction_debug/vis/Test__S03__CoastGuard3__000174__USV_Containership01_C_8__CamLeft.png` |
| 5 | Train/T07/CoastGuard2/000682 | USV_libertyship_C_0 | CamLeft | 259.989 | 248.281 | True | 248.281 | 259.989 | False | `output/z_axis_direction_debug/vis/Train__T07__CoastGuard2__000682__USV_libertyship_C_0__CamLeft.png` |
| 6 | Test/S01/CoastGuard3/000655 | USV_Sailboat01_C_0 | CamFront | 279.761 | 273.623 | True | 273.623 | 279.761 | False | `output/z_axis_direction_debug/vis/Test__S01__CoastGuard3__000655__USV_Sailboat01_C_0__CamFront.png` |
| 7 | Train/T07/CoastGuard1/000642 | USV_queenmarry_C_4 | CamFront | 256.650 | 238.851 | True | 238.851 | 256.650 | False | `output/z_axis_direction_debug/vis/Train__T07__CoastGuard1__000642__USV_queenmarry_C_4__CamFront.png` |
| 8 | Train/T04/CoastGuard3/000541 | USV_fishingboat05_C_0 | CamLeft | 295.624 | 281.232 | True | 281.232 | 295.624 | False | `output/z_axis_direction_debug/vis/Train__T04__CoastGuard3__000541__USV_fishingboat05_C_0__CamLeft.png` |
| 9 | Train/T05/CoastGuard1/000167 | USV_Containership01_C_28 | CamRight | 300.478 | 175.283 | True | 175.283 | 300.478 | False | `output/z_axis_direction_debug/vis/Train__T05__CoastGuard1__000167__USV_Containership01_C_28__CamRight.png` |
| 10 | Train/T02/CoastGuard2/000059 | USV_fishingboat02_C_4 | CamBack | 278.658 | 235.346 | True | 235.346 | 278.658 | False | `output/z_axis_direction_debug/vis/Train__T02__CoastGuard2__000059__USV_fishingboat02_C_4__CamBack.png` |
| 11 | Train/T08/CoastGuard3/000631 | USV_Yacht01_C_0 | CamLeft | 263.311 | 258.851 | True | 258.851 | 263.311 | False | `output/z_axis_direction_debug/vis/Train__T08__CoastGuard3__000631__USV_Yacht01_C_0__CamLeft.png` |
| 12 | Valid/V01/CoastGuard2/000283 | USV_CoastGuard1_C_410 | CamLeft | 234.727 | 225.318 | True | 225.318 | 234.727 | False | `output/z_axis_direction_debug/vis/Valid__V01__CoastGuard2__000283__USV_CoastGuard1_C_410__CamLeft.png` |

- A(top<bottom)=True count: 12/12
- B(top<bottom)=True count: 0/12

## 4) Camera extrinsics z-flip check
- Loader uses a fixed conversion `camera_cv_from_local = [[0,1,0],[0,0,-1],[1,0,0]]` before per-camera extrinsic.
- det(camera_cv_from_local[:3,:3]) = -1.000000.
- The `-1` term (second row, third column) flips local z when mapping to CV y (image-down axis). This is a fixed axis-convention conversion, not a sample-dependent random flip.

## Final Answers
- 当前 radar/ego 坐标系 z 增大方向：**z increases upward** （由 A/B 投影统计主导判断）。
- RouteROI 3D cuboid 建议使用：**`top_z = bottom_z + height`**。
- 是否存在 camera_extrinsics 造成的额外 z 翻转：**存在固定坐标系转换中的符号映射（local z -> CV y 含负号）**，但它是统一定义，不会在样本间额外随机翻转。
- 投影验证统计样本数：**12**（见上表，>=10）。
