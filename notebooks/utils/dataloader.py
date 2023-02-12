import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
from glob import glob
import pandas as pd
import torch.nn.functional as F

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file,img_dir , num_clases=3 ,transform=None, target_transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(glob(os.path.join(self.img_dir, "*.jpg")))

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "imgPath"]
        label = self.df.loc[idx, "label"]
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes = 3)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# def one_hot_encode(data):
#     drop_enc = OneHotEncoder(drop='first').fit(data)
#     drop_enc.categories_
#     drop_enc.transform(data).toarray()
#     return drop_enc
    
    
if __name__ == "__main__":
    train_transforms = transforms.Compose([transforms.Resize((224,224))])
    train_features = CustomImageDataset("./output.csv","../test/",train_transforms)
    dl = DataLoader(train_features,batch_size=2,shuffle=True)
    for image, label in dl:
        print(image.shape)
        print(label)
        break