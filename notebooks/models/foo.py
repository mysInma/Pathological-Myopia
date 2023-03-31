import sys
sys.path.insert(0,"../")
from utils.dataloader import ResnetDataset
from utils.transformations import CustomTransformations
from models.resnet50 import MyopiaClasificationModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
from torch.utils.data import DataLoader
train_features = ResnetDataset("../train.csv","../../test/",transform=CustomTransformations(512))
train_loader = DataLoader(train_features,batch_size=1,shuffle=True)
for img,label in train_loader:
    print("hola")