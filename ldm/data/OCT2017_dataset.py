import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# gray scale img
class OCT2017Base(Dataset):
    def __init__(self, dataset_path="..\\..\\data\\OCT2017", mode="train", sub_dataset_name="NORMAL", size_norm=[228,228], interpolation="bicubic"):
        self.folder_path = os.path.join(dataset_path,mode, sub_dataset_name)
        self.size_norm = size_norm
        self.interpolation = {
                              "bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        # print(self.folder_path)
        file_list = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                tmp = file.split(".")
                if tmp[len(tmp)-1] == "jpeg":
                    file_list.append(os.path.join(root, file))
        self.file_list = file_list
        # print(self.file_list)
        # print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        im = Image.open(file_name)
        if len(im.size) == 2:
            img = np.array(im).astype(np.uint8)
            img3 = np.zeros([img.shape[0],img.shape[1],3]).astype(np.uint8)
            for i in range(3):
                img3[:,:,i] = img
            im = Image.fromarray(img3)

        if self.size_norm is not None:
            im = im.resize((self.size_norm[0], self.size_norm[1]), resample=self.interpolation)

        im_norm = np.array(im).astype(np.uint8)
        im_norm = (im_norm/127.5 - 1.0).astype(np.float32)
        example = {"image": im_norm}
        return example

# demo for loading data
if __name__ == "__main__":
    # demo for using. Define the dataset_path (to OCT2017), and the normalized size, at least.
    train_ds = OCT2017Base(dataset_path="..\\..\\data\\OCT2017", size_norm=[228, 228])
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    for i, ex in enumerate(train_dl):
        input = ex["image"]
        print(i, input.dtype, input.shape)
        input = np.asarray(input)
        input = np.squeeze(input[2,:,:])
        input = (input - np.min(input))/(np.max(input)-np.min(input))*255
        image = Image.fromarray(input.astype(np.uint8))
        image.show()
        break
