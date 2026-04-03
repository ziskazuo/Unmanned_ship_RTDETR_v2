import numpy as np


def _flatten_poly8(segmentation):
    if segmentation is None:
        return None
    if isinstance(segmentation, list) and segmentation and isinstance(
            segmentation[0], list):
        segmentation = segmentation[0]
    poly = np.array(segmentation, dtype=np.float32).reshape(-1)
    if poly.size != 8:
        return None
    return poly


def _safe_unit(vec, fallback):
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def clamp_poly_min_edge(poly, min_edge=2.0):
    """Clamp a quadrilateral's edge lengths by preserving center/orientation.

    Args:
        poly (Sequence[float]): [x1,y1,...,x4,y4]
        min_edge (float): minimum width/height (pixels)
    Returns:
        np.ndarray shape (8,), float32
    """
    min_edge = float(min_edge)
    flat = np.array(poly, dtype=np.float32).reshape(-1)
    if flat.size != 8 or min_edge <= 0:
        return flat

    pts = flat.reshape(4, 2)
    center = np.mean(pts, axis=0).astype(np.float32)

    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    e23 = pts[2] - pts[3]  # same forward direction as e01
    e30 = pts[3] - pts[0]  # same forward direction as e12

    w = 0.5 * (np.linalg.norm(e01) + np.linalg.norm(e23))
    h = 0.5 * (np.linalg.norm(e12) + np.linalg.norm(e30))

    if w <= 1e-6 and h <= 1e-6:
        return flat
    if w >= min_edge and h >= min_edge:
        return flat

    ex = _safe_unit(
        e01 + e23, _safe_unit(e01, np.array([1.0, 0.0], dtype=np.float32)))
    ey_raw = _safe_unit(
        e12 + e30, np.array([-ex[1], ex[0]], dtype=np.float32))
    ey = ey_raw - np.dot(ey_raw, ex) * ex
    ey = _safe_unit(ey, np.array([-ex[1], ex[0]], dtype=np.float32))

    w_new = max(float(w), min_edge)
    h_new = max(float(h), min_edge)

    p0 = center - 0.5 * w_new * ex - 0.5 * h_new * ey
    p1 = center + 0.5 * w_new * ex - 0.5 * h_new * ey
    p2 = center + 0.5 * w_new * ex + 0.5 * h_new * ey
    p3 = center - 0.5 * w_new * ex + 0.5 * h_new * ey
    return np.stack([p0, p1, p2, p3], axis=0).reshape(8).astype(np.float32)


def clamp_segmentation_min_edge(segmentation, min_edge=2.0):
    """Clamp COCO polygon segmentation to min edge and return [[8 floats]]."""
    flat = _flatten_poly8(segmentation)
    if flat is None:
        return segmentation
    clamped = clamp_poly_min_edge(flat, min_edge=min_edge)
    return [clamped.tolist()]


def segmentation_to_poly8(segmentation):
    flat = _flatten_poly8(segmentation)
    if flat is None:
        return None
    return flat


def poly8_to_xyxy(poly8):
    flat = np.array(poly8, dtype=np.float32).reshape(-1)
    if flat.size != 8:
        return None
    xs = flat[0::2]
    ys = flat[1::2]
    return [
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs)),
        float(np.max(ys)),
    ]
