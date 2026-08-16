# Tumor Segmentation -- Multi-Paradigm Comparison

Comparing CNN, Vision Transformer, and Foundation Model for automatic Whole Tumor segmentation in 3D MRI volumes (NIfTI format).

---

## What this project does

Three segmentation models run on the same MRI upload and produce side-by-side results. The goal is not to pick a winner upfront, but to see how differently these three paradigms handle the same task.

| Model | Paradigm |
|---|---|
| SegResNet | CNN |
| Swin-UNETR | Vision Transformer |
| MedSAM | Foundation Model |

The web interface runs fully offline. No data leaves the local machine.

---

## Features

- Parallel inference with all three models on a single NIfTI upload
- Three-view visualization (axial, sagittal, coronal) with red Whole Tumor contour overlay
- Dice Score and Sensitivity computed against ground truth mask
- PDF report generation with patient data, segmentation images, and metrics
- REST API backend (FastAPI) and web frontend (Streamlit)

---

## Architecture

```
Streamlit Frontend
  Upload NIfTI  ->  Results (3 models)  ->  PDF

FastAPI Backend
  main.py  ->  segresnet.py
           ->  swinunetr.py
           ->  medsam.py
  preprocessing.py  .  visualization.py
```

---

## Project structure

```
project/
|-- backend/
|   |-- main.py               FastAPI endpoints
|   |-- segresnet.py          SegResNet inference
|   |-- swinunetr.py          Swin-UNETR inference
|   |-- medsam.py             MedSAM inference
|   |-- preprocessing.py      Z-score normalization, resampling, slice selection
|   `-- visualization.py      Red WT contour, 3-view images
|-- frontend/
|   `-- app.py                Streamlit UI
|-- models/
|   `-- weights/              Place .pth files here (not included in repo)
|-- training/
|   `-- train_segresnet.py    Training script (SegResNet on BraTS Task01)
|-- requirements.txt
|-- install.bat
`-- README.md
```

---

## Dataset

**BraTS Task01** -- Brain Tumor Segmentation

- 484 training cases, NIfTI format (.nii.gz)
- Volume dimensions: 240 x 240 x 155 x 4 (four MRI sequences: FLAIR, T1, T1ce, T2)
- Labels: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- This project segments the Whole Tumor (WT) region only

Download: [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

---

## Model selection

### Why these three?

Four criteria, applied in order:

1. One model per AI paradigm (CNN / Transformer / Foundation Model)
2. Reproducible training and evaluation through MONAI
3. Native 3D processing (MedSAM is a deliberate exception for comparison purposes)
4. Best-documented performance on BraTS within each paradigm


---

## Installation

```bash
git clone https://github.com/faycalbouchama1997-ctrl/ai-powered-tumor-segmentation.git
cd ai-powered-tumor-segmentation
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or run `install.bat` on Windows for a guided setup.

### Model weights

Place the following files in `models/weights/`:

| File | Source |
|---|---|
| `segresnet.pth` | Trained in this project on BraTS Task01 (484 cases) |
| `swinunetr.pth` | [MONAI research-contributions (SwinUNETR)](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/) |
| `medsam_vit_b.pth` | [MedSAM GitHub](https://github.com/bowang-lab/MedSAM) |

---

## Usage

Start backend and frontend in two separate terminals.

**Terminal 1 -- Backend**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 -- Frontend**
```bash
streamlit run frontend/app.py --server.port 8501
```

Open http://localhost:8501, upload a `.nii.gz` file, and click Analyze.

If port 8501 is already in use, pass a different port: `--server.port 8502`.

---

## Evaluation

```bash
python evaluation/evaluate_models.py
```

Output: `evaluation/results.xlsx` with Dice Score and Sensitivity per model and per case.

---

## Results

| Model | Dice Score | Sensitivity | Paradigm |
|---|---|---|---|
| Swin-UNETR | 0.659 | 0.633 | Vision Transformer |
| SegResNet | 0.520 | 0.538 | CNN |
| MedSAM | 0.152 | 0.186 | Foundation Model |

MedSAM's lower score reflects its 2D slice-wise processing. Without 3D context across the volume, it consistently undersegments or misses the tumor region. This is an architectural limitation, not a failure of the model in its intended use case.

---

## License

For academic and research use.

---

## Author

Project developed by Faycal Bouchama.
