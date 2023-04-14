import sys
sys.path.insert(0,"../")
from models.resnet50 import MyopiaClasificationModel
from utils.dataloader import ResnetDataset
from torch.utils.data import DataLoader
import torch
config = {
        "batch_size":1,
        "img_size":1024,
        "num_workers":2,
        "num_classes":3,
        "lr":1e-3
    }
model = MyopiaClasificationModel.load_from_checkpoint("./model-checkpoint/resnet50-epoch=1-train_val_acc=0.71.ckpt")
#model = torch.load("./model-checkpoint/resnet50-epoch=1-train_val_acc=0.71.ckpt")

val_dataset = ResnetDataset("../train_resnet50/val_resnet50.csv","../../train_resnet50/",transform=None)
val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)

model.eval()
import numpy as np 
predicted = np.array([])
target = np.array([])
from tqdm import tqdm
with tqdm(total=72) as pbar:
    for x,y in val_loader:
        with torch.no_grad():
            yhat = model(x)
        y_pred = torch.argmax(yhat, dim=1).cpu().numpy() # predicted labels
        y_true = y.cpu().numpy() # true labels
        predicted = np.append(predicted,y_pred)
        target = np.append(target,y_true)
        pbar.update(1)
np.save("predicted.np",predicted)
np.save("true.np",predicted)