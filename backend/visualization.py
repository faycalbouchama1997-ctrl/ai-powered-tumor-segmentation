import numpy as np
import cv2
from PIL import Image
import io
import base64

OUTPUT_SIZE = 512
MIN_CONTOUR_AREA = 80


def crop_to_organ(gray: np.ndarray, mask: np.ndarray = None, margin: float = 0.05):
    if gray.max() == 0:
        return gray, mask

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, mask

    brain_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(brain_cnt)

    pad_x = max(6, int(w * margin))
    pad_y = max(6, int(h * margin))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray.shape[1], x + w + pad_x)
    y2 = min(gray.shape[0], y + h + pad_y)

    gray_out = gray[y1:y2, x1:x2]
    mask_out = mask[y1:y2, x1:x2] if mask is not None else None
    return gray_out, mask_out


def enhance_contrast(gray_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_u8)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(enhanced, 1.7, blurred, -0.7, 0)


def resize_letterbox(img: np.ndarray, size: int, interp=cv2.INTER_LANCZOS4):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    channels = img.shape[2] if img.ndim == 3 else None
    canvas = np.zeros((size, size, channels), dtype=np.uint8) if channels else np.zeros((size, size), dtype=np.uint8)

    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (top, left, nh, nw)


def draw_tumor_contour(
    gray_slice: np.ndarray,
    mask_slice: np.ndarray,
    contour_thickness: int = 3,
    contour_color: tuple = (255, 0, 0),
    smooth_iterations: int = 2,
    crop: bool = True,
) -> np.ndarray:
    if crop:
        gray_c, mask_c = crop_to_organ(gray_slice, mask_slice)
        if gray_c.size == 0:
            gray_c, mask_c = gray_slice, mask_slice
        if mask_c is None:
            mask_c = mask_slice
    else:
        gray_c, mask_c = gray_slice, mask_slice

    enhanced = enhance_contrast(gray_c.astype(np.uint8))
    gray_big, _ = resize_letterbox(enhanced, OUTPUT_SIZE, cv2.INTER_LANCZOS4)
    mask_u8 = (mask_c > 0).astype(np.uint8) * 255
    mask_big, _ = resize_letterbox(mask_u8, OUTPUT_SIZE, cv2.INTER_NEAREST)
    mask_big = (mask_big > 127).astype(np.uint8) * 255
    img_rgb = cv2.cvtColor(gray_big, cv2.COLOR_GRAY2RGB)

    if not mask_big.any():
        return img_rgb

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for _ in range(smooth_iterations):
        mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel)
        mask_big = cv2.GaussianBlur(mask_big, (7, 7), 2)
        _, mask_big = cv2.threshold(mask_big, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if contours:
        contours = [max(contours, key=cv2.contourArea)]

    if not contours:
        return img_rgb

    overlay = img_rgb.copy()
    cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness + 4, cv2.LINE_AA)
    img_rgb = cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0)
    cv2.drawContours(img_rgb, contours, -1, contour_color, contour_thickness, cv2.LINE_AA)

    return img_rgb


def draw_tumor_contours_multiregion(
    gray_slice: np.ndarray,
    mask_et: np.ndarray,
    mask_wt: np.ndarray,
    mask_tc: np.ndarray,
    contour_thickness: int = 2,
    crop: bool = True,
) -> np.ndarray:
    combined = mask_wt if mask_wt.any() else (mask_tc if mask_tc.any() else mask_et)

    if crop:
        gray_c, _ = crop_to_organ(gray_slice, combined)
        if gray_c.size == 0:
            gray_c = gray_slice

        _, binary = cv2.threshold(gray_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            brain_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(brain_cnt)
            pad_x = max(6, int(w * 0.05))
            pad_y = max(6, int(h * 0.05))
            x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
            x2 = min(gray_slice.shape[1], x + w + pad_x)
            y2 = min(gray_slice.shape[0], y + h + pad_y)
            gray_c = gray_slice[y1:y2, x1:x2]
            mask_et = mask_et[y1:y2, x1:x2]
            mask_wt = mask_wt[y1:y2, x1:x2]
            mask_tc = mask_tc[y1:y2, x1:x2]
    else:
        gray_c = gray_slice

    enhanced = enhance_contrast(gray_c.astype(np.uint8))
    gray_big, _ = resize_letterbox(enhanced, OUTPUT_SIZE, cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(gray_big, cv2.COLOR_GRAY2RGB)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    regions = [
        (mask_wt, (160, 82, 45)),
        (mask_tc, (30, 30, 180)),
        (mask_et, (220, 30, 30)),
    ]

    for msk, color in regions:
        if not msk.any():
            continue
        mask_u8 = (msk > 0).astype(np.uint8) * 255
        mask_big, _ = resize_letterbox(mask_u8, OUTPUT_SIZE, cv2.INTER_NEAREST)
        mask_big = (mask_big > 127).astype(np.uint8) * 255

        for _ in range(2):
            mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel)
            mask_big = cv2.GaussianBlur(mask_big, (7, 7), 2)
            _, mask_big = cv2.threshold(mask_big, 127, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if not cnts:
            continue
        cnts = [max(cnts, key=cv2.contourArea)]

        overlay = img_rgb.copy()
        cv2.drawContours(overlay, cnts, -1, color, contour_thickness + 5, cv2.LINE_AA)
        img_rgb = cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0)
        cv2.drawContours(img_rgb, cnts, -1, color, contour_thickness + 1, cv2.LINE_AA)

    return img_rgb


def array_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def array_to_base64_png(img_rgb: np.ndarray) -> str:
    return base64.b64encode(array_to_png_bytes(img_rgb)).decode("utf-8")


def compute_tumor_volume_cm3(mask_3d: np.ndarray, voxel_mm: tuple = (1.0, 1.0, 1.0)) -> float:
    vox_vol = voxel_mm[0] * voxel_mm[1] * voxel_mm[2]
    return float(mask_3d.sum()) * vox_vol / 1000.0


def generate_report(masks_dict: dict, generation: int, used_fallback: bool,
                    voxel_mm: tuple = (1.0, 1.0, 1.0)) -> dict:
    wt_vol = compute_tumor_volume_cm3(masks_dict["WT"], voxel_mm)
    tc_vol = compute_tumor_volume_cm3(masks_dict["TC"], voxel_mm)
    et_vol = compute_tumor_volume_cm3(masks_dict["ET"], voxel_mm)
    vol_cm3 = wt_vol
    detected = bool(masks_dict["WT"].any())

    model_names = {
        1: "SegResNet (MONAI)",
        3: "Swin-UNETR",
        4: "MedSAM",
    }
    accuracy_notes = {
        1: "CNN · Residual-Blöcke · BraTS 2018 Sieger",
        3: "Transformer · Swin-ViT · globale Kontextinformation",
        4: "Foundation Model · 1,5M medizinische Bilder",
    }

    if detected:
        if vol_cm3 < 5:
            classif = "Kleine Läsion (< 5 cm³)"
        elif vol_cm3 < 20:
            classif = "Mittlere Läsion (5–20 cm³)"
        elif vol_cm3 < 60:
            classif = "Große Läsion (20–60 cm³)"
        else:
            classif = "Sehr große Läsion (> 60 cm³)"
    else:
        classif = "Keine Läsion erkannt"

    return {
        "generation": generation,
        "modele": model_names.get(generation, f"Gen{generation}"),
        "fallback_utilise": used_fallback,
        "tumeur_detectee": detected,
        "volume_wt_cm3": round(wt_vol, 2),
        "volume_tc_cm3": round(tc_vol, 2),
        "volume_et_cm3": round(et_vol, 2),
        "volume_cm3": round(wt_vol, 2),
        "volume_mm3": round(wt_vol * 1000, 1),
        "classification_volume": classif,
        "note_precision": accuracy_notes.get(generation, ""),
        "rapport_narratif": None,
    }
