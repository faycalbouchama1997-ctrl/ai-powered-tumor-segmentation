import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Tuple


def load_nifti(path: Union[str, Path]) -> Tuple[np.ndarray, object]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def normalize_percentile(volume: np.ndarray, pmin: float = 0.5, pmax: float = 99.5) -> np.ndarray:
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume
    p_low = np.percentile(volume[brain_mask], pmin)
    p_high = np.percentile(volume[brain_mask], pmax)
    if p_high == p_low:
        return np.zeros_like(volume)
    clipped = np.clip(volume, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low)


def normalize_zscore(volume: np.ndarray) -> np.ndarray:
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume
    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std()
    if std == 0:
        return np.zeros_like(volume)
    result = np.zeros_like(volume)
    result[brain_mask] = (volume[brain_mask] - mean) / std
    return result


def load_brats_volume(file_path: Union[str, Path]) -> dict:
    data, affine = load_nifti(file_path)

    if data.ndim == 4 and data.shape[-1] == 4:
        flair, t1, t1ce, t2 = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    elif data.ndim == 4 and data.shape[0] == 4:
        flair, t1, t1ce, t2 = data[0], data[1], data[2], data[3]
    elif data.ndim == 4:
        vol = data.mean(axis=-1)
        flair = t1 = t1ce = t2 = vol
    else:
        flair = t1 = t1ce = t2 = data

    return {
        "flair":    normalize_zscore(flair),
        "t1":       normalize_zscore(t1),
        "t1ce":     normalize_zscore(t1ce),
        "t2":       normalize_zscore(t2),
        "t1ce_raw": t1ce,
        "affine":   affine,
        "shape":    t1ce.shape,
    }


def prepare_tensor_monai(volume_dict: dict, target_size: Tuple[int, int, int] = (128, 128, 128)):
    import torch
    import torch.nn.functional as F

    channels = np.stack([
        volume_dict["flair"],
        volume_dict["t1"],
        volume_dict["t1ce"],
        volume_dict["t2"],
    ], axis=0)

    tensor = torch.from_numpy(channels).float().unsqueeze(0)

    if tensor.shape[2:] != target_size:
        tensor = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)

    return tensor


def find_best_slice(t1ce_volume: np.ndarray, axis: int = 2,
                    mask_3d: np.ndarray = None,
                    gaussian_weight: bool = True) -> int:
    n_slices = t1ce_volume.shape[axis]
    lo = int(n_slices * 0.10)
    hi = int(n_slices * 0.90)

    best_idx, best_score = lo, -1
    mid = (lo + hi) / 2.0

    for i in range(lo, hi):
        if axis == 0:
            msk_sl = mask_3d[i, :, :] if mask_3d is not None else None
            t1_sl  = t1ce_volume[i, :, :]
        elif axis == 1:
            msk_sl = mask_3d[:, i, :] if mask_3d is not None else None
            t1_sl  = t1ce_volume[:, i, :]
        else:
            msk_sl = mask_3d[:, :, i] if mask_3d is not None else None
            t1_sl  = t1ce_volume[:, :, i]

        raw_score = float(msk_sl.sum()) if mask_3d is not None else float((t1_sl > 0).sum())

        if gaussian_weight:
            sigma = (hi - lo) / 4.0
            score = raw_score * float(np.exp(-0.5 * ((i - mid) / sigma) ** 2))
        else:
            score = raw_score

        if score > best_score:
            best_score, best_idx = score, i

    return best_idx


def volume_to_display_slice(t1ce_raw: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    if axis == 0:
        sl = t1ce_raw[slice_idx, :, :]
    elif axis == 1:
        sl = t1ce_raw[:, slice_idx, :]
    else:
        sl = t1ce_raw[:, :, slice_idx]

    p1, p99 = np.percentile(sl, 1), np.percentile(sl, 99)
    if p99 == p1:
        return np.zeros(sl.shape, dtype=np.uint8)
    sl_norm = np.clip((sl - p1) / (p99 - p1), 0, 1)
    return (sl_norm * 255).astype(np.uint8)


def mask_slice(mask_3d: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    if axis == 0:
        return mask_3d[slice_idx, :, :]
    elif axis == 1:
        return mask_3d[:, slice_idx, :]
    else:
        return mask_3d[:, :, slice_idx]
