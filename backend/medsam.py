import numpy as np
from pathlib import Path
from typing import Tuple
from scipy import ndimage

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "weights" / "medsam_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"
IMAGE_SIZE = 1024


def is_available() -> bool:
    try:
        import segment_anything
        return WEIGHTS_PATH.exists()
    except ImportError:
        return False


def _detect_brain_mask(t1ce_raw: np.ndarray) -> np.ndarray:
    if not (t1ce_raw > 0).any():
        return np.zeros_like(t1ce_raw, dtype=bool)
    threshold = np.percentile(t1ce_raw[t1ce_raw > 0], 3)
    rough = t1ce_raw > threshold
    rough = ndimage.binary_fill_holes(rough)
    labeled, n = ndimage.label(rough)
    if n == 0:
        return rough
    sizes = [ndimage.sum(rough, labeled, i + 1) for i in range(n)]
    return labeled == (int(np.argmax(sizes)) + 1)


def _approximate_subregions(wt_mask: np.ndarray, t1ce_raw: np.ndarray) -> dict:
    if not wt_mask.any():
        empty = np.zeros_like(t1ce_raw, dtype=bool)
        return {"ET": empty, "WT": wt_mask, "TC": empty}
    brain_voxels = t1ce_raw[wt_mask]
    tc_mask = wt_mask & (t1ce_raw > np.percentile(brain_voxels, 40))
    et_mask = wt_mask & (t1ce_raw > np.percentile(brain_voxels, 75))
    return {"ET": et_mask, "WT": wt_mask, "TC": tc_mask}


def preprocess_slice_medsam(t1ce_slice: np.ndarray) -> np.ndarray:
    import cv2
    p1, p99 = np.percentile(t1ce_slice, 1), np.percentile(t1ce_slice, 99)
    if p99 == p1:
        gray = np.zeros(t1ce_slice.shape, dtype=np.uint8)
    else:
        gray = np.clip((t1ce_slice - p1) / (p99 - p1), 0, 1)
        gray = (gray * 255).astype(np.uint8)
    gray_resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.stack([gray_resized, gray_resized, gray_resized], axis=-1)


def compute_bbox_medsam(t1ce_slice: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    h_orig, w_orig = t1ce_slice.shape
    brain_mask = t1ce_slice > np.percentile(t1ce_slice[t1ce_slice > 0], 5) if (t1ce_slice > 0).any() else np.zeros_like(t1ce_slice, dtype=bool)
    if not brain_mask.any():
        return np.array([0, 0, target_size, target_size])

    brain_pixels = t1ce_slice[brain_mask]
    threshold = np.percentile(brain_pixels, 92)
    bright_mask = (t1ce_slice > threshold) & brain_mask

    if not bright_mask.any():
        return np.array([0, 0, target_size, target_size])

    rows = np.where(np.any(bright_mask, axis=1))[0]
    cols = np.where(np.any(bright_mask, axis=0))[0]
    y_min, y_max = rows[0], rows[-1]
    x_min, x_max = cols[0], cols[-1]

    dy = max(5, int((y_max - y_min) * 0.15))
    dx = max(5, int((x_max - x_min) * 0.15))
    y_min = max(0, y_min - dy)
    y_max = min(h_orig - 1, y_max + dy)
    x_min = max(0, x_min - dx)
    x_max = min(w_orig - 1, x_max + dx)

    scale_y = target_size / h_orig
    scale_x = target_size / w_orig

    return np.array([
        int(x_min * scale_x),
        int(y_min * scale_y),
        int(x_max * scale_x),
        int(y_max * scale_y),
    ])


def predict(volume_dict: dict) -> Tuple[dict, bool]:
    t1ce_raw = volume_dict["t1ce_raw"]
    brain_mask = _detect_brain_mask(t1ce_raw)
    brain_voxels = t1ce_raw[brain_mask]

    if len(brain_voxels) == 0:
        empty = np.zeros_like(t1ce_raw, dtype=bool)
        return {"ET": empty, "WT": empty, "TC": empty}, False

    threshold = np.percentile(brain_voxels, 96)
    candidate = (t1ce_raw > threshold) & brain_mask

    candidate = ndimage.binary_dilation(candidate, iterations=2)
    candidate = ndimage.binary_fill_holes(candidate)
    candidate = ndimage.binary_closing(candidate, iterations=6)
    candidate = ndimage.binary_erosion(candidate, iterations=1)

    labeled, n = ndimage.label(candidate)
    if n == 0:
        empty = np.zeros_like(t1ce_raw, dtype=bool)
        return {"ET": empty, "WT": empty, "TC": empty}, False

    total_brain = max(brain_mask.sum(), 1)
    sizes = np.array([ndimage.sum(candidate, labeled, i + 1) for i in range(n)])

    wt_mask = np.zeros_like(t1ce_raw, dtype=bool)
    for idx in np.argsort(sizes)[::-1]:
        comp = labeled == (idx + 1)
        comp_size = comp.sum()
        fraction = comp_size / total_brain
        if comp_size < 60:
            continue
        if fraction > 0.14:
            continue
        wt_mask |= comp
        if wt_mask.sum() / total_brain > 0.11:
            break

    if not wt_mask.any():
        wt_mask = labeled == (int(np.argmax(sizes)) + 1)

    if wt_mask.any():
        wt_mask = ndimage.binary_fill_holes(wt_mask)

    return _approximate_subregions(wt_mask.astype(bool), t1ce_raw), False
