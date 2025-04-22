import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MIMICCSVImageDataset(Dataset):
    def __init__(self,
                 csv_path,
                 data_root,
                 size=128,
                 interpolation="bicubic",
                 flip_p=0.0  # 通常测试时不做翻转
                 ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.size = size
        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"].split(";")[0]  # 只用第一张图
        img_path = os.path.join(self.data_root, img_path)
        image = Image.open(img_path).convert("L")

        img = np.array(image).astype(np.uint8)
        h, w = img.shape
        crop = min(h, w)
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip(image)

        image = np.array(image).astype(np.float32)
        image = (image / 127.5) - 1.0
        image = np.expand_dims(image, axis=0)

        return {
            "image": image,
            "img_path": row["img_path"],
            "study_id": row["study_id"]
        }

