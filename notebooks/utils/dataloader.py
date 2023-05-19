import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
from glob import glob
import pandas as pd
import torch.nn.functional as F1
import numpy as np

class ResnetDataset(Dataset):
    def __init__(self, annotations_file,img_dir , num_clases=3 ,transform=None, target_transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_clases=num_clases

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "imgPath"]
        label = self.df.loc[idx, "label"]
        label = torch.tensor(label)
        # label = F.one_hot(label, self.num_clases).float()
        image = read_image(img_path).float()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class UNETDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
      self.df = pd.read_csv(annotations_file) 
      self.transform = transform
      self.target_transform = target_transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "imgPath"]
        mask_path = self.df.loc[idx, "maskPath"]
        
        image,mask = self.transform(img_path,mask_path)
        
        return image, mask
    


class VGGDataset(Dataset):
    def __init__(self, annotations_file,img_dir ,transform=None, target_transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df.shape[0]

    # def __getitem__(self, idx):
    #     img_path = self.df.loc[idx, "imgPath"]
    #     x_fovea, y_fovea = list(map(lambda x: float(x) ,self.df.loc[idx, "xy_fovea"].split("/")))
    #     image = read_image(img_path)
    #     if self.transform:
    #         image = self.transform(image)
    #     fovea_loc = torch.tensor([x_fovea, y_fovea]) #Para que se almecene en un formato compatible con Pytorch
               
    #     return image, fovea_loc
    
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "imgPath"]
        x_fovea, y_fovea = list(map(lambda x: float(x) ,self.df.loc[idx, "xy_fovea"].split("/")))
        image = read_image(img_path)
        fovea_loc = torch.tensor([x_fovea, y_fovea])
        if self.transform:
            image_res, fovea_loc = self.transform(image, fovea_loc)
        
        return image_res, fovea_loc
    
    
  


# def one_hot_encode(data):
#     drop_enc = OneHotEncoder(drop='first').fit(data)
#     drop_enc.categories_
#     drop_enc.transform(data).toarray()
#     return drop_enc
    
    
# if __name__ == "__main__":
#     train_transforms = transforms.Compose([transforms.Resize((224,224))])
#     train_features = ResnetDataset("./output.csv","../test/",train_transforms)
#     dl = DataLoader(train_features,batch_size=2,shuffle=True)
#     for image, label in dl:
#         print(image.shape)
#         print(label)
#         break