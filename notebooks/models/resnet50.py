import pytorch_lightning as pl
import os
import math
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.functional import accuracy, auroc 
from torchmetrics import Accuracy
import torchmetrics
import torch


class ResNet50TF(nn.Module):
    def __init__(self, img_size, num_classes=3):
        super().__init__()
        self.resNet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resNet50.fc = nn.Linear(2048,num_classes,bias=True)
        self.baseInputLayers = self.init_input_layer(img_size)
        
        
    def init_input_layer(self,image_size):
        exp_img_size = math.log2(image_size)
        exp_img_size_dest = math.log2(256)
        arch_layers = []
        for _ in range(int(exp_img_size-exp_img_size_dest)):
            arch_layers.append(nn.Conv2d(3,3,(3,3),padding=1,stride=1))
            arch_layers.append(nn.ReLU())
        return nn.Sequential(*arch_layers)
    
    def forward(self,images):
        x = self.baseInputLayers(images)
        x = self.resNet50(x)
        return x


# define the LightningModule
class MyopiaClasificationModel(pl.LightningModule):
    def __init__(self, img_size):
        super().__init__()
        self.model = ResNet50TF(img_size=img_size,num_classes=3)
        self.accuracy = Accuracy(task="multiclass", num_classes=3)
       
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        logits = self(x)
        yhat = F.softmax(logits,dim=1)
        loss = F.cross_entropy(yhat,y)
        acc = self.accuracy(yhat, torch.argmax(y,dim=1))
        #auroc = torchmetrics.functional.auroc(yhat, y, num_classes=3, task='multiclass')
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        #self.log("train_auroc", auroc)
    
        
        return {"loss": loss, "acc": acc}
        #return {"loss": loss, "acc": acc, "auroc":auroc}
    
    

    def training_epoch_end(self, training_step_outputs):
        losses = [x["loss"] for x in training_step_outputs]
        accs =  [x["acc"] for x in training_step_outputs]
        #aurocs =  [x["auroc"] for x in training_step_outputs]
        
        avg_loss = torch.stack(losses).mean()
        train_acc = torch.stack(accs).mean()
        #train_auroc = torch.stack(aurocs).mean()

        
        
        
        self.log("train_loss_ephoc", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc_ephoc", train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("step",self.current_epoch)
        #self.log("train_auroc_ephoc", train_auroc, on_epoch=True, prog_bar=True, logger=True)
        

        
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
