#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 tag 读取一帧：
1) 世界系 GT(AirSim NED) -> 自车系 GT(NED)，保留朝向并补上 box 尺寸(默认 15x5x5m)
2) 单相机：将 3D box 投影到环视 RGB（从 AirSim NED 外参转换到 CV 相机系），用 OpenCV 画线
   - 仅当“目标中心点投影在图像范围内”才绘制与保存 2D 框
   - 保存 2D 框时进行 clip + 小框过滤
3) 可选：保存拼接环视图像与对应 2D GT

坐标/约定：
- AirSim NED：x前 / y右 / z下（右手系），欧拉角 Z-Y-X (yaw->pitch->roll)
- CV 相机系：x右 / y下 / z前
- 3D 框的定义：position 的 (x,y) 是平面中心；z 是盒子的“底面高度”
  因此本地盒子的顶面 z = -H（向上为负），底面 z = 0
"""

from __future__ import annotations
import argparse, sys, json, yaml, math, glob, os
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Optional, Literal

# ===================== 常量/映射 =====================
SENSOR_CODE_MAP = {
    6: "lidar",
    9: "gps",
    10: "imu",
}

# ===================== 基础 IO =====================
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===================== settings 读取 / 解析 =====================
def _find_settings_path():
    # 你可以按需修改：默认当前目录搜 settings.json
    cand = Path("settings.json")
    if cand.is_file():
        return cand
    raise FileNotFoundError("未找到 settings.json，请显式传入路径")

def load_settings(path=None):
    if path is None:
        path = _find_settings_path()
    else:
        path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _sensor_type_of(cfg: dict) -> str:
    """返回 'lidar' | 'imu' | 'gps' | 'other'"""
    st = cfg.get("SensorType", cfg.get("sensor_type"))
    if isinstance(st, int):
        return SENSOR_CODE_MAP.get(st, "other")
    if isinstance(st, str):
        s = st.strip().lower()
        if "lidar" in s:
            return "lidar"
        if "imu" in s:
            return "imu"
        if "gps" in s:
            return "gps"
    return "other"

def parse_layout_from_settings(settings: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    仅根据 settings.json 推断每辆车的相机/雷达/IMU/GPS 名称。
    返回：{ vehicle: {"cameras": [...], "lidars": [...], "imus": [...], "gps": [...]} }
    """
    layout: Dict[str, Dict[str, List[str]]] = {}
    vehicles = settings.get("Vehicles", {}) or {}
    top_cams = settings.get("Cameras", {}) or {}

    # 若仅 1 辆车，则把顶层 Cameras 也分配给这辆车
    default_vehicle_for_top = list(vehicles.keys())[0] if len(vehicles) == 1 else None

    for vname, vcfg in vehicles.items():
        cams: List[str] = []
        lidars: List[str] = []
        imus: List[str] = []
        gpss: List[str] = []

        # 车辆内部 Cameras
        for cname in (vcfg.get("Cameras", {}) or {}).keys():
            cams.append(cname)

        # 车辆内部 Sensors
        for sname, scfg in (vcfg.get("Sensors", {}) or {}).items():
            stype = _sensor_type_of(scfg)
            if stype == "lidar":
                lidars.append(sname)
            elif stype == "imu":
                imus.append(sname)
            elif stype == "gps":
                gpss.append(sname)

        if default_vehicle_for_top == vname and top_cams:
            cams.extend(list(top_cams.keys()))

        layout[vname] = {
            "cameras": sorted(set(cams)),
            "lidars":  sorted(set(lidars)),
            "imus":    sorted(set(imus)),
            "gps":     sorted(set(gpss)),
        }

    if not vehicles and top_cams:
        layout["Vehicle1"] = {
            "cameras": sorted(top_cams.keys()),
            "lidars": [],
            "imus": [],
            "gps": [],
        }

    return layout

# ===================== active.json 读取 =====================
def load_active_boxes(active_path: Path) -> dict:
    """
    读取 active.json，返回 {name: {"L": m, "W": m, "H": m}}
    active.json 中是以厘米为单位（bbox_cm），这里统一换算为米。
    """
    if active_path is None or not Path(active_path).is_file():
        return {}
    data = load_json(active_path)
    dims_map = {}
    # 兼容两种形态：数组 or {"agents":[...]}
    items = data.get("agents") if isinstance(data, dict) and "agents" in data else data
    if not isinstance(items, list):
        return {}

    for ag in items:
        name = ag.get("name")
        box_cm = (ag.get("bbox_cm") or {})
        if not name or not box_cm:
            continue
        try:
            Lm = float(box_cm.get("L", 0.0)) / 100.0
            Wm = float(box_cm.get("W", 0.0)) / 100.0
            Hm = float(box_cm.get("H", 0.0)) / 100.0
            if Lm > 0 and Wm > 0 and Hm > 0:
                dims_map[name.lower()] = {"L": Lm, "W": Wm, "H": Hm}
        except Exception:
            continue
    return dims_map


# ===================== 路径/命名 =====================
def zero_pad_tag(tag, width=4):
    if isinstance(tag, int) or (isinstance(tag, str) and tag.isdigit()):
        return f"{int(tag):0{width}d}"
    return str(tag).zfill(width)

def build_paths(dataset_root, session, tag4, vehicle_key):
    base = Path(dataset_root) / session
    gt_path  = base / "groundtruth" / "gt_under_world" / f"{tag4}_gt_world.yaml"
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    objs = data.get("objects", {}) or {}
    ego_name = [name for name in objs.keys() if vehicle_key in name]
    cam_dir  = base / vehicle_key / "cam"
    lidar_dir= base / vehicle_key / "lidar"
    return {"gt": gt_path, "cam_dir": cam_dir, 'lidar_dir': lidar_dir, 'ego_name':ego_name}

def guess_surround_images(cam_dir, tag4):
    """匹配 {tag4}_cam_{front|right|back|left}_rgb.*"""
    cams = {"Front":"front","Right":"right","Back":"back","Left":"left"}
    out = {k: None for k in cams}
    for cam_key, suffix in cams.items():
        matches = sorted(glob.glob(str(Path(cam_dir) / f"{tag4}_cam_{suffix}_rgb.*")))
        if matches:
            out[cam_key] = Path(matches[0])
    return out

# ===================== 线性代数 / 位姿（AirSim NED） =====================
def rot_x(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def rot_y(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[ c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rot_z(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[ c,-s,0],[ s,c,0],[0,0,1]], dtype=float)

def se3_from_euler_ned(x, y, z, roll, pitch, yaw):
    """AirSim NED (x前/y右/z下)，Z-Y-X 组合：Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3,3] = [x,y,z]
    return T

def se3_inv(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4, dtype=float)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

def K_from_fov(W, H, fov_deg):
    """水平 FOV + 分辨率 → 内参（方像素），匹配 CV 相机系(x右/y下/z前)"""
    fx = 0.5 * W / math.tan(math.radians(fov_deg)/2.0)
    fy = fx
    cx, cy = W/2.0, H/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=float)

def base_name_from_gt(name: str) -> str:
    """
    提取 name 第一个 '_' 左右两部分，用于和 active.json 的 name 对齐。
    转为小写，避免大小写不一致导致匹配失败。
    """
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}".lower()
    return name.lower()

# ===================== 世界 GT(NED) -> 自车 GT(NED) =====================
def world_to_ego_with_orientation_airsim_dimsmap(gt_world, ego_name, dims_map: dict,
                                                 default_dims=(15.0,5.0,5.0)):
    """
    和 world_to_ego_with_orientation_airsim 类似，但 box 尺寸来自 dims_map[name]（单位：米）。
    若某对象不在 dims_map，则回退到 default_dims。
    """
    objs = gt_world.get("objects", {})
    if ego_name not in objs:
        raise KeyError(f"{ego_name} 不在 GT.objects")
    ego = objs[ego_name]

    ex,ey,ez = ego["position"]["x"], ego["position"]["y"], ego["position"]["z"]
    er = ego.get("euler_deg",{}).get("roll",0.0)
    ep = ego.get("euler_deg",{}).get("pitch",0.0)
    eyw= ego.get("euler_deg",{}).get("yaw", 0.0)
    T_w_ego = se3_from_euler_ned(ex,ey,ez, er,ep,eyw)
    T_ego_w = se3_inv(T_w_ego)

    out = {"ego_name": ego_name, "unit": "ego(NED)", "objects": {}}
    for name, obj in objs.items():
        ox,oy,oz = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]
        orr = obj.get("euler_deg",{}).get("roll",  0.0)
        opi = obj.get("euler_deg",{}).get("pitch", 0.0)
        oya = obj.get("euler_deg",{}).get("yaw",   0.0)

        T_w_obj  = se3_from_euler_ned(ox,oy,oz, orr,opi,oya)
        T_ego_obj= T_ego_w @ T_w_obj
        Pe = T_ego_obj[:3,3]; Re = T_ego_obj[:3,:3]

        # 提取 Z-Y-X Euler（deg）
        sy = -Re[2,0]; sy = max(min(sy, 1.0), -1.0)
        pitch = math.asin(sy)
        if abs(sy) < 0.999999:
            yaw  = math.atan2(Re[1,0], Re[0,0])
            roll = math.atan2(Re[2,1], Re[2,2])
        else:
            yaw  = math.atan2(-Re[0,1], Re[1,1])
            roll = 0.0

        # 从 dims_map 取尺寸，默认为 default_dims
        base_name = base_name_from_gt(name)
        dm = dims_map.get(base_name)
        if dm:
            Lm, Wm, Hm = float(dm["L"]), float(dm["W"]), float(dm["H"])
        else:
            Lm, Wm, Hm = default_dims

        out["objects"][name] = {
            "position":  {"x": float(Pe[0]), "y": float(Pe[1]), "z": float(Pe[2])},
            "euler_deg": {"roll": float(math.degrees(roll)),
                          "pitch": float(math.degrees(pitch)),
                          "yaw":   float(math.degrees(yaw))},
            "box":       {"dims_m": {"L": Lm, "W": Wm, "H": Hm}},
        }
    return out

# ===================== 相机外参（AirSim NED -> CV 相机系） =====================
def make_T_camCV_ego_airsim(cam_cfg):
    """
    输出：K, (W,H), T_camCV_ego（自车点→CV相机点）
    推导：
      T_ego_camNED (给定) → 取逆 T_camNED_ego
      基底变换 NED→CV:  M = [[0,1,0],[0,0,1],[1,0,0]]
      R_camCV_ego = M @ R_camNED_ego,  t_camCV_ego = M @ t_camNED_ego
    """
    cs = (cam_cfg.get("CaptureSettings") or [{}])[0]
    W = int(cs.get("Width",  1000)); H = int(cs.get("Height", 1000))
    fov = float(cs.get("FOV_Degrees", 90.0))
    K = K_from_fov(W, H, fov)

    x = cam_cfg.get("X",0.0); y = cam_cfg.get("Y",0.0); z = cam_cfg.get("Z",0.0)
    r = cam_cfg.get("Roll",0.0); p = cam_cfg.get("Pitch",0.0); yw = cam_cfg.get("Yaw",0.0)

    T_ego_camNED = se3_from_euler_ned(x,y,z, r,p,yw)
    T_camNED_ego = se3_inv(T_ego_camNED)

    M = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]], dtype=float)  # NED -> CV

    R_camCV_ego = M @ T_camNED_ego[:3,:3]
    t_camCV_ego = M @ T_camNED_ego[:3, 3]

    T_camCV_ego = np.eye(4, dtype=float)
    T_camCV_ego[:3,:3] = R_camCV_ego
    T_camCV_ego[:3, 3] = t_camCV_ego
    return K, (W,H), T_camCV_ego

# ===================== 3D Box（XY中心，Z底面；顶面为 -H） =====================
BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

def box_corners_local_xy_center_z_base(L, W, H):
    """
    局部原点：(0,0,0) = 底面中心
    - X/Y：以中心对称，±L/2、±W/2
    - Z：  底面=0，顶面=-H   （NED 坐标 z 向下为正，要“向上”取负）
    返回 3x8（列向量）的角点：先顶面(0..3)，再底面(4..7)
    """
    l2, w2 = L/2.0, W/2.0
    corners = np.array([
        [ l2,  w2, -H], [ l2, -w2, -H], [-l2, -w2, -H], [-l2,  w2, -H],  # top
        [ l2,  w2,  0], [ l2, -w2,  0], [-l2, -w2,  0], [-l2,  w2,  0],  # bottom
    ], dtype=float).T  # 3x8
    return corners

# ===================== 投影（CV 相机系） =====================
def project_points_cv(K, X_camCV):
    x, y, z = X_camCV[0], X_camCV[1], X_camCV[2]
    zsafe = np.where(z <= 1e-6, np.nan, z)
    u = K[0,0]*(x/zsafe) + K[0,2]
    v = K[1,1]*(y/zsafe) + K[1,2]
    return u, v, z

def draw_box_cv(img, uv_pts, color=(0,0,255), thickness=2):
    H, W = img.shape[:2]
    def ok(pt):
        u,v = pt
        return np.isfinite(u) and np.isfinite(v) and (-W<=u<=2*W) and (-H<=v<=2*H)
    for i,j in BOX_EDGES:
        p0, p1 = uv_pts[i], uv_pts[j]
        if ok(p0) and ok(p1):
            cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                          (int(round(p1[0])), int(round(p1[1]))),
                          color, thickness, lineType=cv2.LINE_AA)

# ======= RGB 拆分后的两个核心函数 =======
def cam_boxes_to_yaml(cam_cfg, ego_gt, ego_name, out_yaml_2d, out_yaml_3d,
                      img_size=None, min_box=5, K_set=None, size_set=None, T_camCV_ego=None):
    """
    只负责计算并保存：
      - 2D 框 YAML: {name: {xmin,ymin,xmax,ymax}}
      - 3D（投影后的8个角点）YAML: {name: {"corners_2d": [[u,v] * 8]}}
    不读取/写入任何图像文件；K/外参可在循环外预先计算后传入以避免重复。
    """
    # 取得 K, 尺寸, 外参
    if K_set is None or size_set is None or T_camCV_ego is None:
        K_set, size_set, T_camCV_ego = make_T_camCV_ego_airsim(cam_cfg)
    W_set, H_set = size_set
    if img_size is None:
        W_img, H_img = W_set, H_set
        K = K_set
    else:
        W_img, H_img = img_size
        K = K_set.copy()
        sx, sy = W_img/float(W_set), H_img/float(H_set)
        K[0,0]*=sx; K[1,1]*=sy; K[0,2]*=sx; K[1,2]*=sy

    boxes_2d = {}
    boxes_3d = {}
    for name, o in ego_gt["objects"].items():
        if name == ego_name:
            continue
        center = np.array([[o["position"]["x"]],[o["position"]["y"]],[o["position"]["z"]]], dtype=float)
        Xc_c = (T_camCV_ego @ np.vstack([center, [[1.0]]]))[:3,:]
        if Xc_c[2,0] <= 1e-6:
            continue
        u_c, v_c, _ = project_points_cv(K, Xc_c)
        if not (0 <= u_c[0] < W_img and 0 <= v_c[0] < H_img):
            continue

        L=float(o["box"]["dims_m"]["L"]); Wb=float(o["box"]["dims_m"]["W"]); Hb=float(o["box"]["dims_m"]["H"])
        er=float(o["euler_deg"]["roll"]); ep=float(o["euler_deg"]["pitch"]); ey=float(o["euler_deg"]["yaw"])
        corners_local = box_corners_local_xy_center_z_base(L,Wb,Hb)
        R_obj_ego = rot_z(ey) @ rot_y(ep) @ rot_x(er)
        corners_ego = (R_obj_ego @ corners_local) + center  # (3,8)
        Xc = (T_camCV_ego @ np.vstack([corners_ego, np.ones((1,8), dtype=float)]))[:3,:]
        u,v,z = project_points_cv(K, Xc)
        valid = np.isfinite(u) & np.isfinite(v) & (z>1e-6)
        if not np.any(valid):
            continue
        # 2D 框
        us, vs = u[valid], v[valid]
        xmin,xmax = float(np.min(us)), float(np.max(us))
        ymin,ymax = float(np.min(vs)), float(np.max(vs))
        if (xmax-xmin)>=min_box and (ymax-ymin)>=min_box:
            boxes_2d[name] = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            # 3D：投影后的8角点（即 3D 结构在图上的线框）
            corners_uv = np.stack([u, v], axis=1).tolist()  # len=8, 含 inf 的点上面已过滤到框外，不影响保存
            boxes_3d[name] = {"corners_2d": corners_uv}

    Path(out_yaml_2d).parent.mkdir(parents=True, exist_ok=True)
    save_yaml(boxes_2d, out_yaml_2d)
    Path(out_yaml_3d).parent.mkdir(parents=True, exist_ok=True)
    save_yaml(boxes_3d, out_yaml_3d)
    return boxes_2d, boxes_3d

def cam_draw_from_yaml(img, boxes2d_yaml, boxes3d_yaml=None, color_2d=(0,0,255), color_3d=(0,255,0), thickness=2):
    """
    只负责绘制：给定一张图像(ndarray)与 YAML 路径，叠加 2D 框与(可选)3D线框，返回图像。
    """
    import yaml
    out = img.copy()
    boxes2d = {}
    if boxes2d_yaml:
        with open(boxes2d_yaml, "r", encoding="utf-8") as f:
            boxes2d = yaml.safe_load(f) or {}
        for name,b in boxes2d.items():
            pt1 = (int(round(b["xmin"])), int(round(b["ymin"])))
            pt2 = (int(round(b["xmax"])), int(round(b["ymax"])))
            cv2.rectangle(out, pt1, pt2, color_2d, thickness, lineType=cv2.LINE_AA)
            cv2.putText(out, name, (pt1[0], max(0, pt1[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_2d, 1, lineType=cv2.LINE_AA)

    if boxes3d_yaml:
        with open(boxes3d_yaml, "r", encoding="utf-8") as f:
            boxes3d = yaml.safe_load(f) or {}
        EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for name, item in boxes3d.items():
            pts = item.get("corners_2d", [])
            if not pts or len(pts) != 8:
                continue
            for i,j in EDGES:
                u0,v0 = pts[i]; u1,v1 = pts[j]
                if np.isfinite(u0) and np.isfinite(v0) and np.isfinite(u1) and np.isfinite(v1):
                    cv2.line(out, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))),
                             color_3d, thickness, lineType=cv2.LINE_AA)
                    
            # ===== 在左上角顶点处标注名称 =====
            u, v = pts[0]  # 第一个角点（左上角）
            if np.isfinite(u) and np.isfinite(v):
                cv2.putText(out, name, (int(u), int(max(0, v - 4))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_3d, 1, lineType=cv2.LINE_AA)
            
    return out

# ===================== 保存 ego GT（封装） =====================
def save_ego_gt(gt_world, ego_name, out_path, active_path: Path = None,
                default_dims=(15.0,5.0,5.0)):
    """
    读取 active.json（若提供）以覆盖每个对象的 box 尺寸（米），然后保存 ego 系 GT。
    """
    dims_map = load_active_boxes(active_path) if active_path else {}
    ego_gt = world_to_ego_with_orientation_airsim_dimsmap(
        gt_world, ego_name, dims_map, default_dims=default_dims
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml(ego_gt, out_path)
    return ego_gt


# ===================== 保存环视拼接（可选） =====================
def save_surround_view(img_map, out_path, boxes_per_cam,
                       order=("Left","Front","Right","Back"),
                       color=(0,255,0), thickness=2):
    """
    拼接环视大图并绘制2D框，同时保存对应的yaml。
    - img_map: {cam_name: 图像路径}
    - boxes_per_cam: {cam_name: {obj_name: {xmin,ymin,xmax,ymax}}}
    """
    imgs, widths, heights = [], [], []
    for cam in order:
        if cam in img_map and img_map[cam] and Path(img_map[cam]).is_file():
            im = cv2.imread(str(img_map[cam]), cv2.IMREAD_COLOR)
        else:
            im = np.zeros((512,512,3), np.uint8)
        imgs.append(im)
        heights.append(im.shape[0]); widths.append(im.shape[1])

    H = max(heights); total_W = sum(widths)
    canvas = np.zeros((H, total_W, 3), np.uint8)

    offset = 0
    all_boxes = {}
    for cam, im in zip(order, imgs):
        h, w = im.shape[:2]
        canvas[:h, offset:offset+w] = im

        if cam in boxes_per_cam:
            for obj, box in boxes_per_cam[cam].items():
                xmin = int(round(box["xmin"] + offset))
                xmax = int(round(box["xmax"] + offset))
                ymin = int(round(box["ymin"]))
                ymax = int(round(box["ymax"]))

                # 保存全局坐标
                all_boxes[f"{cam}_{obj}"] = {
                    "xmin": float(xmin), "ymin": float(ymin),
                    "xmax": float(xmax), "ymax": float(ymax)
                }

                # === 新增：直接绘制到环视图上 ===
                cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax),
                              color=color, thickness=thickness)
                cv2.putText(canvas, f"{cam}_{obj}",
                            (xmin, max(0, ymin-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color, 1, cv2.LINE_AA)

        offset += w

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)

    gt_path = Path(out_path).with_suffix(".yaml")
    save_yaml(all_boxes, gt_path)
    return out_path, gt_path

def make_T_lidar_ego_airsim(lidar_cfg):
    """
    从 settings.json 的 lidar 配置构造自车->雷达坐标系的变换
    AirSim 坐标: NED (x前,y右,z下)
    """
    x = lidar_cfg.get("X", 0.0)
    y = lidar_cfg.get("Y", 0.0)
    z = lidar_cfg.get("Z", 0.0)
    r = lidar_cfg.get("Roll", 0.0)
    p = lidar_cfg.get("Pitch", 0.0)
    yw= lidar_cfg.get("Yaw", 0.0)

    # 自车->雷达(NED)
    T_ego_lidar = se3_from_euler_ned(x, y, z, r, p, yw)
    return T_ego_lidar

# ======= PPI 拆分后的两个核心函数 =======
def ppi_boxes_to_yaml(ego_gt, T_lidar_ego, lidar_range, ppi_res, ego_name, out_yaml):
    """
    只负责计算并保存 PPI 平面上的多边形（像素坐标）到 YAML。
    不依赖任何图像/路径；T_lidar_ego、lidar_range、ppi_res 请在循环外预先确定后传入。
    """
    import numpy as np, math
    scale = (ppi_res/2.0) / max(float(lidar_range), 1e-6)
    cx = cy = ppi_res/2.0
    boxes_ppi = {}
    for name, o in ego_gt.get("objects", {}).items():
        if ego_name and name == ego_name:
            continue
        # ego -> lidar
        P_ego = np.array([o["position"]["x"], o["position"]["y"], o["position"]["z"], 1.0], dtype=float)
        P_lidar = (T_lidar_ego @ P_ego)[:3]
        # P_lidar = P_ego
        x, y = float(P_lidar[0]), float(P_lidar[1])
        if (x*x + y*y) > float(lidar_range)**2:
            continue
        L  = float(o["box"]["dims_m"]["L"]); Wb = float(o["box"]["dims_m"]["W"])
        yaw= float(o["euler_deg"]["yaw"])
        l2, w2 = L/2.0, Wb/2.0
        corners = np.array([[ l2,  w2],
                            [ l2, -w2],
                            [-l2, -w2],
                            [-l2,  w2]], dtype=float).T
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        R = np.array([[c,-s],[s,c]], dtype=float)
        corners_xy = (R @ corners) + np.array([[x],[y]], dtype=float)
        us = cx + corners_xy[1,:]*scale
        vs = cy - corners_xy[0,:]*scale
        pts = np.vstack([us,vs]).T.astype(int)
        boxes_ppi[name] = {"poly": pts.tolist()}
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    save_yaml(boxes_ppi, out_yaml)
    return boxes_ppi

def ppi_draw_from_yaml(ppi_img, boxes_yaml_path, color=(0,0,255), thickness=2):
    """
    只负责绘制：输入一张 PPI 图像 (np.ndarray) 与 YAML 文件路径（第一步输出），
    将多边形画到该图像并返回图像（不负责保存和读取图像路径）。
    """
    # import cv2, yaml, numpy as np
    if ppi_img is None:
        raise ValueError("ppi_img 不能为空（应为 np.ndarray 图像）。")
    with open(boxes_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = ppi_img.copy()
    for name, item in data.items():
        pts = np.asarray(item.get("poly", []), dtype=np.int32)
        if pts.size == 0: 
            continue
        pts = pts.reshape(-1,1,2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        c = pts.reshape(-1,2).mean(axis=0)
        cv2.putText(out, name, (int(c[0]), int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out

# ======= PCD 拆分后的两个核心函数（仅 YAML / 仅可视化） =======

def pcd_boxes_to_yaml(ego_gt: dict, T_lidar_ego: np.ndarray, ego_name: str, out_yaml: Path):
    """
    只负责计算并保存：每个目标在【LIDAR系】下的 8 个 3D 角点。
    YAML 结构：{ name: {"corners_lidar": [[x,y,z] * 8]} }
    不读取/写入任何点云文件；后续可视化函数再根据这些角点采样出线段点。
    """
    def _box_corners_local_xy_center_z_base(L, W, H):
        l2, w2 = L/2.0, W/2.0
        return np.array([
            [ l2,  w2, -H], [ l2, -w2, -H], [-l2, -w2, -H], [-l2,  w2, -H],  # top
            [ l2,  w2,  0], [ l2, -w2,  0], [-l2, -w2,  0], [-l2,  w2,  0],  # bottom
        ], dtype=float).T  # (3,8)

    boxes3d = {}
    for name, o in (ego_gt.get("objects", {}) or {}).items():
        if ego_name and name == ego_name:
            continue
        L = float(o["box"]["dims_m"]["L"]); W = float(o["box"]["dims_m"]["W"]); H = float(o["box"]["dims_m"]["H"])
        x = float(o["position"]["x"]);       y = float(o["position"]["y"]);       z = float(o["position"]["z"])
        r = float(o["euler_deg"]["roll"]);   p = float(o["euler_deg"]["pitch"]);  yw= float(o["euler_deg"]["yaw"])
        # 8角点：EGO → LIDAR
        corners_local = _box_corners_local_xy_center_z_base(L,W,H)                               # (3,8)
        T_ego_obj = se3_from_euler_ned(x,y,z, r,p,yw)                                            # 4×4
        corners_ego = (T_ego_obj   @ np.vstack([corners_local, np.ones((1,8), dtype=float)]))[:3,:]
        corners_lid = (T_lidar_ego @ np.vstack([corners_ego,  np.ones((1,8), dtype=float)]))[:3,:]
        boxes3d[name] = {"corners_lidar": corners_lid.T.tolist()}  # (8,3) 列表

    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    save_yaml(boxes3d, out_yaml)
    return boxes3d


def pcd_overlay_from_yaml(pcd_in: Path, boxes_yaml: Path, pcd_out: Path,
                          pts_per_edge: int = 80,
                          color_pts=(200,200,200), color_box=(255,0,0)):
    """
    只负责可视化：读取点云 pcd_in（ASCII XYZ），读取 YAML（LIDAR系 8角点），
    将 3D 框采样成红色线段点，与原始点合并后写出 pcd_out（ASCII xyzrgb）。
    """
    # --- 工具函数（与原 run 内部逻辑一致） ---
    def _pack_rgb_uint32(r, g, b):
        return (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)

    def _write_pcd_xyzrgb_ascii(path: Path, xyz: np.ndarray, rgb_uint32: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        N = int(xyz.shape[0])
        header = [
            "# .PCD v0.7 - Point Cloud Data file format","VERSION 0.7",
            "FIELDS x y z rgb","SIZE 4 4 4 4","TYPE F F F F","COUNT 1 1 1 1",
            f"WIDTH {N}","HEIGHT 1","VIEWPOINT 0 0 0 1 0 0 0",f"POINTS {N}","DATA ascii",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            rgbf = rgb_uint32.view(np.float32)
            for (x, y, z), c in zip(xyz.astype(np.float32), rgbf.astype(np.float32)):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {c:.9e}\n")

    def _read_pcd_xyz_ascii(path: Path) -> np.ndarray:
        if not Path(path).is_file():
            return np.empty((0,3), np.float32)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        data_idx, fields = None, None
        for i, ln in enumerate(lines):
            if ln.startswith("FIELDS"): fields = ln.split()[1:]
            if ln.startswith("DATA"):
                if "ascii" not in ln:
                    return np.empty((0,3), np.float32)
                data_idx = i + 1; break
        if data_idx is None or fields is None:
            return np.empty((0,3), np.float32)
        try:
            ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
        except ValueError:
            return np.empty((0,3), np.float32)
        pts = []
        for ln in lines[data_idx:]:
            if not ln: continue
            tk = ln.split()
            if len(tk) <= max(ix,iy,iz): continue
            try:
                pts.append((float(tk[ix]), float(tk[iy]), float(tk[iz])))
            except:
                pass
        return np.asarray(pts, dtype=np.float32) if pts else np.empty((0,3), np.float32)

    EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    # 1) 读原始点
    xyz_pts = _read_pcd_xyz_ascii(pcd_in)
    rgb_pts = np.full((xyz_pts.shape[0],), _pack_rgb_uint32(*color_pts), dtype=np.uint32)

    # 2) 读 YAML，采样线框
    with open(boxes_yaml, "r", encoding="utf-8") as f:
        box_dict = yaml.safe_load(f) or {}

    xyz_box = []
    for name, item in box_dict.items():
        corners = np.asarray(item.get("corners_lidar", []), dtype=np.float32)  # (8,3)
        if corners.shape != (8,3):
            continue
        for i,j in EDGES:
            p0, p1 = corners[i], corners[j]
            t = np.linspace(0.0, 1.0, int(max(2, pts_per_edge)), dtype=np.float32)[:,None]
            seg = p0*(1.0 - t) + p1*t
            xyz_box.append(seg)
    if xyz_box:
        xyz_box = np.vstack(xyz_box)
        rgb_box = np.full((xyz_box.shape[0],), _pack_rgb_uint32(*color_box), dtype=np.uint32)
    else:
        xyz_box = np.empty((0,3), np.float32)
        rgb_box = np.empty((0,), np.uint32)

    # 3) 合并写出
    xyz_all = np.vstack([xyz_pts, xyz_box]) if xyz_box.size else xyz_pts
    rgb_all = np.hstack([rgb_pts,  rgb_box]) if xyz_box.size else rgb_pts
    _write_pcd_xyzrgb_ascii(pcd_out, xyz_all, rgb_all)
    return pcd_out

# ===================== run_for_tag（精简组织） =====================
def run_for_tag(settings_path, dataset_root, session,
                tag, out_dir,
                vehicle_key="CoastGuard2", surround_order = ["Front","Right","Back","Left"],
                active_path: Path = None,
                default_dims_m=(150.0,50.0,50.0),
                save_surround=False):
    tag4 = zero_pad_tag(tag, 4)
    paths = build_paths(dataset_root, session, tag4, vehicle_key)

    gt_world_path, cam_dir, lidar_dir, ego_name = paths["gt"], paths["cam_dir"], paths["lidar_dir"], paths["ego_name"]

    if not Path(gt_world_path).is_file():
        raise FileNotFoundError(f"GT 不存在: {gt_world_path}")
    if not Path(cam_dir).is_dir():
        raise FileNotFoundError(f"相机目录不存在: {cam_dir}")

    # 1) 读取 settings / layout / 取环视相机
    settings = load_settings(settings_path)
    layout = parse_layout_from_settings(settings)
    if vehicle_key not in layout:
        raise KeyError(f"settings 中无 {vehicle_key}，可选={list(layout.keys())}")


    # 2) 保存新 GT（世界->自车）,ego_gt自车坐标系下的3d box
    gt_world = load_yaml(gt_world_path)
    out_gt_path = Path(out_dir) / "groundtruth" / f"{Path(gt_world_path).stem}_{ego_name}.yaml"
    ego_gt = save_ego_gt(gt_world, ego_name, out_gt_path,
                         active_path=active_path,
                         default_dims=default_dims_m)

    # 3) 单相机绘制与 2D 框保存（中心点在图像内才保留）
    v_cams_cfg = (settings.get("Vehicles", {}).get(vehicle_key, {}).get("Cameras") or {})
    surround = {k: v_cams_cfg[k] for k in surround_order if k in v_cams_cfg}
    cam_precomp = {}
    for cam_name, cam_cfg in surround.items():
        K_set, size_set, T_camCV_ego = make_T_camCV_ego_airsim(cam_cfg)
        cam_precomp[cam_name] = (K_set, size_set, T_camCV_ego)

    img_map = guess_surround_images(cam_dir, tag4)
    out_images, boxes_per_cam = {}, {}

    for cam_name, cam_cfg in surround.items():
        in_img  = img_map.get(cam_name)
        # 确定图像尺寸用于 2D 缩放
        cs = (cam_cfg.get("CaptureSettings") or [{}])[0]
        W0 = int(cs.get("Width",1000)); H0 = int(cs.get("Height",1000))

        # YAML 输出路径
        out_dir_sur = Path(out_dir) / "surround"
        box2d_yaml = out_dir_sur / f"{tag4}_cam_{cam_name.lower()}_rgb_2dbox.yaml"
        box3d_yaml = out_dir_sur / f"{tag4}_cam_{cam_name.lower()}_rgb_3dbox.yaml"

        K_set, size_set, T_camCV_ego = cam_precomp[cam_name]
        # 1) 只保存 YAML（2D+3D） 自车坐标系投影至相机系2d与3d框
        cam_boxes_to_yaml(cam_cfg, ego_gt, ego_name, box2d_yaml, box3d_yaml,
                          img_size=(W0,H0), min_box=5,
                          K_set=K_set, size_set=size_set, T_camCV_ego=T_camCV_ego)
        boxes_per_cam[cam_name] = load_yaml(box2d_yaml)

        # 2) 绘制（传入图像对象）
        if in_img and Path(in_img).is_file():
            img0 = cv2.imread(str(in_img), cv2.IMREAD_COLOR)
            img_with = cam_draw_from_yaml(img0, box2d_yaml, box3d_yaml, color_2d=(0,0,255), color_3d=(0,255,0), thickness=2)
            out_img = out_dir_sur / f"{tag4}_cam_{cam_name.lower()}_rgb_box.png"
            cv2.imwrite(str(out_img), img_with)
            out_images[cam_name] = out_img


    # 4) 可选：保存环视拼接
    out = {"gt_ego": out_gt_path, "imgs": out_images}
    if save_surround:
        surround_img_path = Path(out_dir) / "surround" / f"{tag4}_surround_rgb.png"
        surround_img, surround_gt = save_surround_view(img_map, surround_img_path, boxes_per_cam,
                                                       order=["Left","Front","Right","Back"])
        out["surround_img"] = surround_img
        out["surround_gt"]  = surround_gt

    
    # 5) 雷达 PPI (BEV)：将不变项放到循环外，循环内先 YAML、后绘制
    v_sensors_cfg = (settings.get("Vehicles", {}).get(vehicle_key, {}).get("Sensors") or {})
    lidar_cfg = None
    for sname, scfg in v_sensors_cfg.items():
        if _sensor_type_of(scfg) == "lidar":
            lidar_cfg = scfg
            break

    if lidar_cfg:
        # 循环外：固定量
        lidar_range = float(lidar_cfg.get("Range", 100.0))
        T_ego_lidar = make_T_lidar_ego_airsim(lidar_cfg)
        T_lidar_ego = se3_inv(T_ego_lidar)
        # 如有底图，外部一次性读取为图像（如果 run_for_tag 外管理更好，这里仍做兜底）
        ppi_img = None
        ppi_candidates = list(Path(lidar_dir).glob(f"{tag4}_ppi.*")) or list(Path(lidar_dir).glob(f"{tag4}_lidars_ppi.*"))
        # ppi_candidates = list(Path(cam_dir).glob(f"{tag4}_cam_topview_seg.*"))
        if ppi_candidates:
            tmp = cv2.imread(str(ppi_candidates[0]), cv2.IMREAD_COLOR)
            if tmp is not None and tmp.shape[0] == tmp.shape[1]:
                ppi_img = tmp

        # 先输出 YAML（仅框）
        ppi_yaml_path = Path(out_dir) / "lidar_ppi" / f"{tag4}_ppi_2dbox.yaml"
        Path(ppi_yaml_path).parent.mkdir(parents=True, exist_ok=True)
        ## 自车坐标系投影至ppi平面2d框
        ppi_boxes_to_yaml(ego_gt, T_lidar_ego, lidar_range, 1000, ego_name, ppi_yaml_path)

        # 再绘制（用图像数组）
        if ppi_img is None:
            ppi_img = np.zeros((1000,1000,3), np.uint8)
        ppi_with = ppi_draw_from_yaml(ppi_img, ppi_yaml_path, color=(0,0,255), thickness=2)
        ppi_out_path = Path(out_dir) / "lidar_ppi" / f"{tag4}_ppi_box.png"
        cv2.imwrite(str(ppi_out_path), ppi_with)

        out["ppi_img"] = ppi_out_path
        out["ppi_gt"]  = ppi_yaml_path

        # --- 仅保存 LIDAR 3D 框（LIDAR系 8角点） ---
        pcd_yaml_path = Path(out_dir) / "lidar_ppi" / f"{tag4}_pcd_3dbox.yaml"
        # 自车坐标系投影至LIDAR系3d框
        pcd_boxes_to_yaml(ego_gt, T_lidar_ego, ego_name, pcd_yaml_path)

        # --- 仅可视化：读入同帧点云，叠加线框后导出新的 PCD ---
        pcd_in  = Path(lidar_dir) / f"{tag4}_lidars_lidar.pcd"   # 采集阶段保存的原始点云（ASCII）
        pcd_out = Path(out_dir) / "lidar_ppi" / f"{tag4}_pcd_gt.pcd"
        pcd_overlay_from_yaml(pcd_in, pcd_yaml_path, pcd_out,
                              pts_per_edge=80, color_pts=(200,200,200), color_box=(255,0,0))

        out["pcd_yaml"]   = pcd_yaml_path
        out["pcd_overlay"] = pcd_out

    else:
        print("[WARN] 未找到 LiDAR 配置，跳过 PPI")
    return out


# ===================== run_for_tag（精简组织） =====================
def vis_for_tag(settings_path, dataset_root, session,
                tag, out_dir,
                vehicle_key="CoastGuard1", surround_order = ["Front","Right","Back","Left"],
                active_path: Path = None,
                default_dims_m=(150.0,50.0,50.0),
                save_surround=False):
    tag4 = zero_pad_tag(tag, 4)
    paths = build_paths(dataset_root, session, tag4, vehicle_key)
    cam_dir, lidar_dir, ego_name =  paths["cam_dir"], paths["lidar_dir"], paths["ego_name"]

    # if not Path(gt_world_path).is_file():
    #     raise FileNotFoundError(f"GT 不存在: {gt_world_path}")
    if not Path(cam_dir).is_dir():
        raise FileNotFoundError(f"相机目录不存在: {cam_dir}")

    # 1) 读取 settings / layout / 取环视相机
    settings = load_settings(settings_path)
    layout = parse_layout_from_settings(settings)
    if vehicle_key not in layout:
        raise KeyError(f"settings 中无 {vehicle_key}，可选={list(layout.keys())}")


    # 2) 保存新 GT（世界->自车）
    gt_yaml_path = dataset_root / session / "groundtruth" / "gt_under_ego" / f"{tag4}_gt_ego.yaml"
    ego_gt = load_yaml(gt_yaml_path)
    # out_gt_path = Path(out_dir) / "groundtruth" / f"{Path(gt_world_path).stem}_{ego_name}.yaml"
    # ego_gt = save_ego_gt(gt_world, ego_name, out_gt_path,
    #                      active_path=active_path,
    #                      default_dims=default_dims_m)

    # 3) 单相机绘制与 2D 框保存（中心点在图像内才保留）
    v_cams_cfg = (settings.get("Vehicles", {}).get(vehicle_key, {}).get("Cameras") or {})
    surround = {k: v_cams_cfg[k] for k in surround_order if k in v_cams_cfg}
    cam_precomp = {}
    for cam_name, cam_cfg in surround.items():
        K_set, size_set, T_camCV_ego = make_T_camCV_ego_airsim(cam_cfg)
        cam_precomp[cam_name] = (K_set, size_set, T_camCV_ego)

    img_map = guess_surround_images(cam_dir, tag4)
    out_images, boxes_per_cam = {}, {}

    for cam_name, cam_cfg in surround.items():
        in_img  = img_map.get(cam_name)


        # # # YAML 输出路径
        out_dir_sur = Path(out_dir) / "surround"
        out_dir_sur.mkdir(parents=True, exist_ok=True)

        box2d_yaml = dataset_root / session / "groundtruth" / "surround" / f"{tag4}_cam_{cam_name.lower()}_rgb_2dbox.yaml"
        box3d_yaml = dataset_root / session / "groundtruth" / "surround" / f"{tag4}_cam_{cam_name.lower()}_rgb_3dbox.yaml"

        # 2) 绘制（传入图像对象）
        if in_img and Path(in_img).is_file():
            img0 = cv2.imread(str(in_img), cv2.IMREAD_COLOR)
            img_with = cam_draw_from_yaml(img0, box2d_yaml, None, color_2d=(0,0,255), color_3d=(0,0,255), thickness=2)
            out_img = out_dir_sur / f"{tag4}_cam_{cam_name.lower()}_rgb_box.png"
            cv2.imwrite(str(out_img), img_with)
            out_images[cam_name] = out_img


    # 4) 可选：保存环视拼接
    out = {"gt_ego": gt_yaml_path, "imgs": out_images}

    # 5) 雷达 PPI (BEV)：将不变项放到循环外，循环内先 YAML、后绘制
    v_sensors_cfg = (settings.get("Vehicles", {}).get(vehicle_key, {}).get("Sensors") or {})
    lidar_cfg = None
    for sname, scfg in v_sensors_cfg.items():
        if _sensor_type_of(scfg) == "lidar":
            lidar_cfg = scfg
            break

    if lidar_cfg:
        # 循环外：固定量
        lidar_range = float(lidar_cfg.get("Range", 100.0))
        T_ego_lidar = make_T_lidar_ego_airsim(lidar_cfg)
        T_lidar_ego = se3_inv(T_ego_lidar)
        # 如有底图，外部一次性读取为图像（如果 run_for_tag 外管理更好，这里仍做兜底）
        ppi_img = None
        # ppi_candidates = list(Path(lidar_dir).glob(f"{tag4}_ppi.*")) or list(Path(lidar_dir).glob(f"{tag4}_lidars_ppi.*"))
        # ppi_candidates = list(Path(cam_dir).glob(f"{tag4}_cam_topview_seg.*"))
        ppi_candidates = list(Path(cam_dir).glob(f"{tag4}_cam_topview_bev_bw.*"))
        # ppi_candidates = list(Path(lidar_dir).glob(f"{tag4}_lidars_ppi.*"))

        if ppi_candidates:
            tmp = cv2.imread(str(ppi_candidates[0]), cv2.IMREAD_COLOR)
            if tmp is not None and tmp.shape[0] == tmp.shape[1]:
                ppi_img = tmp

        # 先输出 YAML（仅框）
        ppi_yaml_path = dataset_root / session / "groundtruth" / "lidar_ppi" / f"{tag4}_ppi_2dbox.yaml"
        
        # ppi_boxes_to_yaml(ego_gt, T_lidar_ego, lidar_range, 1000, ego_name, ppi_yaml_path)

        # 再绘制（用图像数组）
        if ppi_img is None:
            ppi_img = np.zeros((1000,1000,3), np.uint8)
        ppi_with = ppi_draw_from_yaml(ppi_img, ppi_yaml_path, color=(0,0,255), thickness=1)
        ppi_out_path = Path(out_dir) / "lidar_ppi" / f"{tag4}_ppi_box.png"
        ppi_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ppi_out_path), ppi_with)

        out["ppi_img"] = ppi_out_path
        out["ppi_gt"]  = ppi_yaml_path

        # --- 仅保存 LIDAR 3D 框（LIDAR系 8角点） ---
        pcd_yaml_path = Path(out_dir) / "lidar_ppi" / f"{tag4}_pcd_3dbox.yaml"
        pcd_boxes_to_yaml(ego_gt, T_lidar_ego, ego_name, pcd_yaml_path)

        # --- 仅可视化：读入同帧点云，叠加线框后导出新的 PCD ---
        pcd_in  = Path(lidar_dir) / f"{tag4}_lidars_lidar.pcd"   # 采集阶段保存的原始点云（ASCII）
        pcd_out = Path(out_dir) / "lidar_ppi" / f"{tag4}_pcd_gt.pcd"
        pcd_overlay_from_yaml(pcd_in, pcd_yaml_path, pcd_out,
                              pts_per_edge=80, color_pts=(200,200,200), color_box=(255,0,0))

        out["pcd_yaml"]   = pcd_yaml_path
        out["pcd_overlay"] = pcd_out

    else:
        print("[WARN] 未找到 LiDAR 配置，跳过 PPI")

    ## 6) 额外保存拼接环视图像与GT
    order = ["Left", "Front", "Right", "Back"]
    imgs = [cv2.imread(str(Path(out_dir_sur) / f"{tag4}_cam_{c.lower()}_rgb_box.png"), cv2.IMREAD_COLOR) for c in order]
    imgs.append(cv2.imread(str(ppi_out_path), cv2.IMREAD_COLOR))

    concat = cv2.hconcat(imgs)

    save_dir = Path(out_dir) / "vis_gt"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{tag4}_surround_plus_ppi.png"
    cv2.imwrite(str(out_path), concat)

    return out


# ===================== CLI =====================
def parse_args():

    ap = argparse.ArgumentParser(description="按 tag 读取一帧 -> 自车系 GT(NED) -> OpenCV 投影到环视 RGB")

    # ap.add_argument("--settings", default=r"C:\STUDY\UN_poject\dataset\USV_dataset2025\2025-10-15_01-42-24/settings.json")
    # ap.add_argument("--active", default=r"C:\STUDY\UN_poject\dataset\USV_dataset2025\2025-10-15_01-42-24/active.json")
    # ap.add_argument("--ego_name", default=r"USV_CoastGuard2_C_112", help="采集船只名称（必须存在于 GT.objects 中）")
    # ap.add_argument("--out_dir", default=r"C:\STUDY\UN_poject\dataset\USV_dataset2025\2025-10-15_01-42-24/output_box2img", help="输出目录")
    ap.add_argument("--dataset_root", default=r"C:\STUDY\UN_poject\dataset\USV_dataset2025_v2", help="数据集根目录（例如 dataset2025_9_21）")
    ap.add_argument("--session", default=r"2025-10-18_22-26-46", help="会话子目录（例如 2025-09-22_23-09-36）")
    # ap.add_argument("--tag", default=1, help="帧序号（'0001' 或 1 均可）")
    ap.add_argument("--vehicle_key", default="CoastGuard1", help="settings.Vehicles 中的键名（默认 CoastGuard1）")
    ap.add_argument("--save_surround", default=True, help="是否额外保存拼接环视图像与GT")
    return ap.parse_args()

def main():
    args = parse_args()
    settings_path = Path(args.dataset_root) / Path(args.session) / "settings.json"
    dataset_root  = Path(args.dataset_root)
    out_dir       = Path(args.dataset_root) / Path(args.session) / "output_vis"
    active_path = Path(args.dataset_root) / "active.json"
    # active_path = None
    tag = 1

    for i in range(227, 228):
        tag = i
        print(f"[INFO] 处理 Tag={tag} ...")
        outputs = vis_for_tag(
            settings_path=settings_path,
            dataset_root=dataset_root,
            session=args.session,
            tag=tag,
            out_dir=out_dir,
            vehicle_key=args.vehicle_key,
            active_path=active_path,
            default_dims_m=(15,5,5),
            save_surround=args.save_surround,
        )

    print("[OK] 生成完成：")


if __name__ == "__main__":
    main()

