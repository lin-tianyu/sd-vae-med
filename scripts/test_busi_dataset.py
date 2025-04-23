import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ldm.data.busi import BUSIDataset

busi_data_root = "D:\JHU\jhu_labs\\25spring\CSSR\project\Dataset_BUSI\Dataset_BUSI_with_GT"

test_dataset = BUSIDataset(data_root=busi_data_root, size=256)

sample = test_dataset[0]
img = sample["image"]
plt.imshow((img + 1) / 2, cmap="gray")
plt.title("test_dataset[0]['image']")
plt.axis("off")
plt.show()

dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    images = batch["image"]
    plt.imshow((images[0] + 1) / 2, cmap="gray")
    plt.title("First image from DataLoader batch")
    plt.axis("off")
    plt.show()
    break
