import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "weights" / "swinunetr.pth"
TARGET_SIZE = (128, 128, 128)
THRESHOLD = 0.5
CHANNEL_MAP = {"ET": 0, "WT": 1, "TC": 2}


def build_model() -> "torch.nn.Module":
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
    )


def is_available() -> bool:
    return WEIGHTS_PATH.exists()


def load_model(device: torch.device) -> Optional["torch.nn.Module"]:
    if not is_available():
        return None
    model = build_model()
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def predict(volume_dict: dict) -> Tuple[dict, bool]:
    from backend.preprocessing import prepare_tensor_monai

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if model is None:
        raise RuntimeError("Swin-UNETR weights not found. Place swinunetr.pth in models/weights/.")

    tensor = prepare_tensor_monai(volume_dict, TARGET_SIZE).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)

    masks_dict = {}
    for name, ch in CHANNEL_MAP.items():
        prob_ch = probs[0, ch].cpu().numpy()
        small_mask = (prob_ch > THRESHOLD).astype(np.float32)
        t = torch.from_numpy(small_mask).unsqueeze(0).unsqueeze(0)
        full_mask = (
            F.interpolate(t, size=volume_dict["shape"], mode="trilinear", align_corners=False)
            .squeeze().numpy() > 0.5
        )
        masks_dict[name] = full_mask

    return masks_dict, False
