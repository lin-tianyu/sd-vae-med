import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BUSIDataset(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.size = size

        self.image_paths = []
        for subfolder in ["benign", "malignant", "normal"]:
            full_path = os.path.join(data_root, subfolder)
            for fname in os.listdir(full_path):
                if fname.endswith('.png') and "_mask" not in fname:
                    self.image_paths.append(os.path.join(full_path, fname))

        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": self.image_paths,
        }

        self.interpolation = {
            "linear": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"]).convert("L")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        # resize
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
            
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example
