import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss

DATA_DIR    = Path(__file__).parent.parent / "dataset" / "Task01_BrainTumour"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "weights" / "segresnet.pth"
TARGET_SIZE = (128, 128, 128)
EPOCHS      = 20
LR          = 1e-4
BATCH_SIZE  = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    return SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        norm=("group", {"num_groups": 8}),
    )


def build_label(seg: np.ndarray) -> np.ndarray:
    et = (seg == 4).astype(np.float32)
    wt = (np.isin(seg, [1, 2, 4])).astype(np.float32)
    tc = (np.isin(seg, [1, 4])).astype(np.float32)
    return np.stack([et, wt, tc], axis=0)


class BraTSDataset(Dataset):
    def __init__(self, image_paths, label_paths, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import nibabel as nib

        img_nib = nib.load(str(self.image_paths[idx]))
        seg_nib = nib.load(str(self.label_paths[idx]))

        img = img_nib.get_fdata(dtype=np.float32)
        seg = seg_nib.get_fdata(dtype=np.float32)

        if img.ndim == 4 and img.shape[-1] == 4:
            img = np.transpose(img, (3, 0, 1, 2))
        elif img.ndim == 4 and img.shape[0] == 4:
            pass
        else:
            img = np.stack([img] * 4, axis=0)

        for ch in range(4):
            brain = img[ch] > 0
            if brain.sum() > 0:
                m, s = img[ch][brain].mean(), img[ch][brain].std()
                if s > 0:
                    img[ch] = np.where(brain, (img[ch] - m) / s, 0.0)

        label = build_label(seg)

        img_t   = torch.from_numpy(img).unsqueeze(0).float()
        label_t = torch.from_numpy(label).unsqueeze(0).float()

        img_t   = F.interpolate(img_t,   size=TARGET_SIZE, mode="trilinear", align_corners=False)
        label_t = F.interpolate(label_t, size=TARGET_SIZE, mode="nearest")

        img_t   = img_t.squeeze(0)
        label_t = label_t.squeeze(0)

        if self.augment:
            for ax in [0, 1, 2]:
                if torch.rand(1).item() < 0.5:
                    img_t   = torch.flip(img_t,   dims=[ax + 1])
                    label_t = torch.flip(label_t, dims=[ax + 1])
            scale = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
            img_t = img_t * scale

        return img_t, label_t


def get_data_paths():
    images_dir = DATA_DIR / "imagesTr"
    labels_dir = DATA_DIR / "labelsTr"

    dataset_json = DATA_DIR / "dataset.json"
    if dataset_json.exists():
        with open(dataset_json) as f:
            meta = json.load(f)
        cases = meta.get("training", [])
        image_paths = [DATA_DIR / c["image"].lstrip("./") for c in cases]
        label_paths = [DATA_DIR / c["label"].lstrip("./") for c in cases]
    else:
        image_paths = sorted([p for p in images_dir.glob("*.nii.gz") if not p.name.startswith("._")])
        label_paths = sorted([p for p in labels_dir.glob("*.nii.gz") if not p.name.startswith("._")])

    return image_paths, label_paths


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    image_paths, label_paths = get_data_paths()
    assert len(image_paths) > 0, f"No data found in {DATA_DIR}"
    print(f"Cases found: {len(image_paths)}")

    train_ds     = BraTSDataset(image_paths, label_paths, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model     = build_model().to(DEVICE)
    criterion = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(imgs)
            loss   = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"  Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, OUTPUT_PATH)

    print(f"Weights saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
