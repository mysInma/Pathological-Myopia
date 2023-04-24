import pytorch_lightning as pl
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import Accuracy   
from torch.nn import Conv2d, MaxPool2d, Upsample, ReLU, BatchNorm2d, Sequential
from torchmetrics import Accuracy  
import math 


from torchmetrics.classification import BinaryRecall, BinaryAUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


class ResUNET(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size=img_size 
        self.loops = int(math.log2(img_size)-math.log2(64))
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def encoder_residualBlock(self,input_channels,mid_channels,x0):
        c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
        c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
        r1 = ReLU()
        r2 = ReLU()
        x = c1(x0)
        x = r1(x)
        x = c2(x)
        x = x + x0
        return r2(x)

    def decoder_residualBlock(self,input_channels,mid_channels,x0):
        r1 = ReLU()
        r2 = ReLU()
        c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
        c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
        skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
        x = c1(x0)
        x = r1(x)
        x = c2(x)
        x = skip(x0) + x
        x = r2(x)
        x = Upsample(scale_factor=2,mode="bilinear",align_corners=True)(x)
        x = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)(x)
        return x

    def convBlock(self,input_channels,output_channels,x0):
        x2 = Conv2d(input_channels,output_channels,kernel_size=3,padding=1)(x0)
        x2 = BatchNorm2d(output_channels)(x2)
        return ReLU()(x2)


    def build_encoder(self):
      # Almacenamos los bloques de convoluciones, el bloque residual no lo incluimos
      base = 2
      res = []
      for i in range(0,self.loops):
        if i ==0:
          res.append(Sequential(*[Conv2d(3,base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU(),
                                  Conv2d(base**(6+i),base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU()                                  
                                  ]))
        else:
          res.append(Sequential(*[Conv2d(base**(6+i-1),base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU(),
                                  Conv2d(base**(6+i),base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU()
                                  ]))
      return res

    def build_decoder(self):
      res = []
      for i in range(self.loops-1):
        res.append(Sequential(*[Conv2d(self.img_size//(2**i),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
                                BatchNorm2d(self.img_size//(2**(i+1))),
                                ReLU(),
                                Conv2d(self.img_size//(2**(i+1)),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
                                BatchNorm2d(self.img_size//(2**(i+1))),
                                ReLU()
                                ]))
      return res
    
    def forward(self,x):
        base = 2
        # Empezamos haciendo la bajada de la U
        x = self.encoder[0](x)
        x = self.encoder_residualBlock(base**6,base**6,x)
        x1 = MaxPool2d(2)(x)
        # (feature sin maxpool, feature despues del maxpool)
        features_encoder = [(x,x1)]
        for idx,layer in enumerate(self.encoder[1:]):
            x = features_encoder[-1][1]
            x = layer(x)
            x = self.encoder_residualBlock(base**(6+idx+1),base**(6+idx+1),x)
            x1 = MaxPool2d(2)(x)
            features_encoder.append((x,x1))
            
        
        # Intermedio

        x = self.convBlock(self.img_size//2,self.img_size,features_encoder[-1][1])
        x = self.convBlock(self.img_size,self.img_size,x)

        x = self.encoder_residualBlock(self.img_size,self.img_size,x)

        x1 = MaxPool2d(2)(x)

        x1 = self.convBlock(self.img_size,self.img_size*2,x1)
        x1 = self.convBlock(self.img_size*2,self.img_size*2,x1)
        

        x1 = self.decoder_residualBlock(self.img_size*2,self.img_size,x1)
        

        x = torch.cat([x,x1],dim=1)
        

        x = self.convBlock(self.img_size*2,self.img_size,x)
        x = self.convBlock(self.img_size,self.img_size,x)
        

        x = self.decoder_residualBlock(self.img_size,self.img_size//2,x)
        

        # Decoder TO-DO
        for idx,layer in enumerate(self.decoder):
          x1 = features_encoder[-1-idx][0]
          x = torch.cat([x1,x],dim=1)
          x = layer(x)
          x = self.decoder_residualBlock(self.img_size//(2**(idx+1)),self.img_size//(2**(idx+2)),x)

        x1 = features_encoder[0][0]
        x = torch.cat([x1,x],dim=1)
        x = self.convBlock(128,64,x)
        x = self.convBlock(64,1,x)

        return x
     
# define the LightningModule
class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = ResUNET(img_size=self.hparams.img_size)
        self.accuracy = Accuracy(task="binary")
        self.auc =  BinaryAUROC()
        self.recall = BinaryRecall(average="macro") 
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
      x, y = batch
      x = x.to(self.device)
      y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
      
      self.log("train_loss_step", loss,prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_acc_step", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_auroc_step", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_recall_step",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
    
      return loss
    
    def training_epoch_end(self, training_step_outputs):
        self.log("step",self.current_epoch)
        
        
    def validation_step(self, batch, batch_idx):
      x,y = batch
      x = x.to(self.device)
      y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
        
      self.log("train_val_loss", loss,prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_val_acc", self.accuracy(logits, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_val_auroc", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_val_recall",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)

      return loss
  
    def validation_epoch_end(self, validation_step_outputs):
      self.log("step",self.current_epoch)
  
      
    def test_step(self,batch,batch_idx):
      x,y = batch
      x = x.to(self.device)
      y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
      self.log("train_test_loss", loss,prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_test_acc", self.accuracy(logits, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_test_auroc", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_test_recall",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)

      return loss
      
    def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
      return optimizer
  
if __name__ == '__main__':
    # model = ResUNET(1024)
    # with torch.no_grad():
    #     for i in range(5):
    #       model(torch.rand(1,3,1024,1024))
    
    
    
    import sys
    sys.path.insert(0,"../")
    from utils.dataloader import UNETDataset
    from utils.transformations import CustomTransformationResUnet
    from models.resunet import ResUNET
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from torchmetrics.functional import accuracy
    import torch
    from torch.utils.data import DataLoader
    
    config = {
        "batch_size":4,
        "img_size":512,
        "num_workers":4,
        "lr":1e-3,
    }
    
    
    pl.seed_everything(42,workers=True)
    # train_features = UNETDataset("../train_unet/Unet_train.csv","../../train_unet/",transform=CustomTransformationResUnet(config["img_size"]))
    # train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    # val_dataset = UNETDataset("../train_unet/Unet_val.csv","../../train_unet/",transform=CustomTransformationResUnet(config["img_size"]))
    # val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)
   
    train_features = UNETDataset("../test_prueba/Unet_train.csv","../../test_prueba/",CustomTransformationResUnet(config["img_size"]))
    train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    val_dataset = UNETDataset("../test_prueba/Unet_val.csv","../../test_prueba/",CustomTransformationResUnet(config["img_size"]))
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
                   ModelCheckpoint(dirpath="./model-checkpoint-resUNET/",\
                    filename="resnet50H-{epoch}-{train_val_acc:.2f}",
                    save_top_k=2,
                    monitor="train_val_loss")],
        log_every_n_steps=1,
        # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt"
    )
    #test_loader = train_loader

    miopia_model = SegmentationModel(config)
    trainer.fit(miopia_model, train_loader,val_loader)
