from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ========== Dataset: 画像とマスクのペア ==========
class ImageMaskDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        size=512,
        image_suffix=".jpg",
        mask_suffix=".png",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        assert self.image_dir.exists()
        assert self.mask_dir.exists()
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.files = sorted([p.stem for p in self.image_dir.glob("*")])
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(self.image_dir / (name + self.image_suffix)).convert("RGB")
        mask = Image.open(self.mask_dir / (name + self.mask_suffix))
        mask.putpalette(color_map())
        return self.transform(img), self.mask_transform(mask.convert("RGB"))


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
