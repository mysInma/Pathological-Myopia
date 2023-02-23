import pytorch_lightning as pl
import os
import math
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy   
from torchmetrics.classification import MulticlassAUROC, MulticlassRecall
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
        self.auc = MulticlassAUROC(num_classes=3)
        self.recall = MulticlassRecall(num_classes=3, average="macro")
       
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        logits = self(x)
        yhat = F.softmax(logits,dim=1)
        loss = F.cross_entropy(logits,y)
        acc = self.auc(yhat, y)
        recall = self.recall(torch.argmax(yhat,dim=1),y)
        auc = self.auc(yhat, y)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss_step", loss,prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_acc_step", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_auroc_step", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_recall_step",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
    
        return loss
        # return {"loss": loss, "acc": acc}
        # return {"loss": loss, "acc": acc, "auroc":auc, "recall":recall}
    
    

    def training_epoch_end(self, training_step_outputs):
    #     losses = [x["loss"] for x in training_step_outputs]
    #     accs =  [x["acc"] for x in training_step_outputs]
    #     aurocs =  [x["auroc"] for x in training_step_outputs]
    #     recalls = [x["recall"] for x in training_step_outputs]
        
    #     avg_loss = torch.stack(losses).mean()
    #     train_acc = torch.stack(accs).mean()
    #     train_auroc = torch.stack(aurocs).mean()
    #     train_recall = torch.stack(recalls).mean()
        
    #     self.log("train_loss_ephoc", avg_loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_acc_ephoc", train_acc, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_auroc_ephoc", train_auroc, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_recall_epoch",train_recall,on_epoch=True,on_step=False)
        self.log("step",self.current_epoch)
        

        
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
img_size=512
    
if __name__ == '__main__':
    import sys
    sys.path.insert(0,"../")
    from utils.dataloader import CustomImageDataset
    from utils.transformations import CustomTransformations
    from models.resnet50 import MyopiaClasificationModel
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from torchmetrics.functional import accuracy, auroc
    import torch
    from torch.utils.data import DataLoader
    train_features = CustomImageDataset("../train.csv","../../test/",transform=CustomTransformations(img_size))
    train_loader = DataLoader(train_features,batch_size=3,num_workers=3,shuffle=True)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=10,
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=4,
        enable_checkpointing=False
    )

    miopia_model = MyopiaClasificationModel(img_size)
    trainer.fit(miopia_model, train_loader)
