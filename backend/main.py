import sys
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.preprocessing import (
    load_brats_volume, find_best_slice,
    volume_to_display_slice, mask_slice,
)
from backend.visualization import draw_tumor_contour, array_to_base64_png, generate_report

app = FastAPI(title="TumorSeg API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/models/status")
def models_status():
    import backend.segresnet as g1
    import backend.swinunetr as g3
    import backend.medsam as g4

    return {
        "gen1": g1.is_available(),
        "gen3": g3.is_available(),
        "gen4": g4.is_available(),
    }


@app.post("/analyze/{generation}")
async def analyze(generation: int, file: UploadFile = File(...)):
    if generation not in (1, 3, 4):
        raise HTTPException(400, "Modell muss 1, 3 oder 4 sein.")

    fname = file.filename or "upload.nii.gz"
    suffix = ".nii.gz" if fname.endswith(".nii.gz") else Path(fname).suffix or ".nii"
    tmp = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        tmp.write_bytes(await file.read())
        volume_dict = load_brats_volume(tmp)
        t1ce_raw = volume_dict["t1ce_raw"]

        used_fallback = False
        if generation == 1:
            import backend.segresnet as g
            mask_3d, used_fallback = g.predict(volume_dict)
        elif generation == 3:
            import backend.swinunetr as g
            mask_3d, used_fallback = g.predict(volume_dict)
        elif generation == 4:
            import backend.medsam as g
            mask_3d, used_fallback = g.predict(volume_dict)

        views = {}
        for axis, label in [(2, "axial"), (0, "sagittal"), (1, "coronal")]:
            idx = find_best_slice(t1ce_raw, axis=axis, mask_3d=mask_3d["WT"])
            gray = volume_to_display_slice(t1ce_raw, idx, axis=axis)
            msk = mask_slice(mask_3d["WT"], idx, axis=axis)
            rgb = draw_tumor_contour(gray, msk)
            views[label] = {"image_base64": array_to_base64_png(rgb), "slice_index": idx}

        report = generate_report(mask_3d, generation, used_fallback)

        return JSONResponse({
            "views": views,
            "image_base64": views["axial"]["image_base64"],
            "slice_index": views["axial"]["slice_index"],
            "report": report,
            "success": True,
        })

    except Exception as e:
        raise HTTPException(500, f"Fehler: {e}")
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
