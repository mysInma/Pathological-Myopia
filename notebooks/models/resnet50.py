import pytorch_lightning as pl
import os
import math
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy   
from torchmetrics.classification import MulticlassAUROC, MulticlassRecall
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
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
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = ResNet50TF(img_size=self.hparams.img_size,num_classes=self.hparams.num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.auc = MulticlassAUROC(num_classes=self.hparams.num_classes)
        self.recall = MulticlassRecall(num_classes=self.hparams.num_classes, average="macro")
       
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        logits = self(x)
        
        yhat = F.softmax(logits,dim=1)
        loss = F.cross_entropy(logits,y)    
        
        # l1 penalization applied
        indices = torch.where(y == 0)
        if torch.any(torch.argmax(yhat,dim=1)[indices]>0):
            penalty = self.hparams.l1_lambda * sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + penalty
        
        
            
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
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        x, y = batch
        logits = self(x)
        yhat = F.softmax(logits,dim=1)
        loss = F.cross_entropy(logits,y)
        
        indices = torch.where(y == 0)
        if torch.any(torch.argmax(yhat,dim=1)[indices]>0):
            penalty = self.hparams.l1_lambda * sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + penalty
        
        self.log("train_val_loss", loss,prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_val_acc", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_val_auroc", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_val_recall",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
    
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log("step",self.current_epoch)
    
    def test_step(self,batch,batch_idx):
        x,y = batch
        x, y = batch
        logits = self(x)
        yhat = F.softmax(logits,dim=1)
        loss = F.cross_entropy(logits,y)
        self.log("train_test_loss", loss,prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_test_acc", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_test_auroc", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
        self.log("train_test_recall",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
    
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
if __name__ == '__main__':
    import sys
    sys.path.insert(0,"../")
    from utils.dataloader import ResnetDataset
    from utils.transformations import CustomTransformations
    from models.resnet50 import MyopiaClasificationModel
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from torchmetrics.functional import accuracy, auroc
    import torch
    from torch.utils.data import DataLoader
    
    config = {
        "batch_size":4,
        "img_size":512,
        "num_workers":4,
        "num_classes":2,
        "lr":1e-3,
        "l1_lambda":1e-5
    }
    
    
    pl.seed_everything(42,workers=True)
    train_features = ResnetDataset("../train_resnet50/resnet_train.csv","../../train_resnet50/",transform=CustomTransformations(config["img_size"]))
    train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    val_dataset = ResnetDataset("../train_resnet50/resnet_val.csv","../../train_resnet50/",transform=CustomTransformations(config["img_size"]))
    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)

    # Initialize a trainer
    trainer = Trainer(
        accumulate_grad_batches=32, # acumula los gradientes de los primeros 4 batches
        #deterministic=True,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=1000,
        callbacks=[TQDMProgressBar(),
                   EarlyStopping(monitor="train_val_loss",mode="min",patience=3),
                   ModelCheckpoint(dirpath="./model-checkpoint/",\
                    filename="resnet50H-{epoch}-{train_val_acc:.2f}",
                    save_top_k=2,
                    monitor="train_val_loss")],
        log_every_n_steps=1,
        # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt"
    )
    #test_loader = train_loader

    miopia_model = MyopiaClasificationModel(config)
    trainer.fit(miopia_model, train_loader,val_loader)
    #trainer.test(miopia_model,test_loader)
