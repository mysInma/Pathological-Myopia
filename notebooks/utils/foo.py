import os
import torch
from torch.utils.data import  DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
from glob import glob
import pandas as pd
import torch.nn.functional as F
from createCSV import vggCSV
from dataloader import VGGDataset

from transformations import CustomTransformationVgg

train_transforms = transforms.Compose([transforms.Resize((1334,1334))])
train_features = VGGDataset("../train_vgg/VGG_train.csv","../../train_vgg/",CustomTransformationVgg(224))
print(train_features.df.shape[0])
dl = DataLoader(train_features,batch_size=2,shuffle=True)
for image, fovea_loc in dl:
    print(image.shape)
    print(fovea_loc.shape) #Esto me coge un tensor con x e y 
    break