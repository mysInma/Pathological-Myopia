import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
from glob import glob
import pandas as pd
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file,img_dir ,transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(glob(os.path.join(self.img_dir, "*.jpg")))

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image
if __name__ == "__main__":
    train_transforms = transforms.Compose([transforms.Resize((224,224))])
    train_features = CustomImageDataset("./output.csv","./dataset_prueba/",train_transforms)
    # print(f"Feature batch shape: {train_features.size()}")
    # img = train_features[0].squeeze()
    # plt.imshow(img, cmap="gray")
    # plt.show()
    dl = DataLoader(train_features,batch_size=2,shuffle=True)
    for image in dl:
        print(image.shape)
        break
