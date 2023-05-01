import pytorch_lightning as pl
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import Accuracy   
from torch.nn import Conv2d, MaxPool2d, Upsample, ReLU, BatchNorm2d, Sequential
from torchmetrics import Accuracy  
import math 
import gc

from torchmetrics.classification import BinaryRecall, BinaryAUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
# total_memory = torch.cuda.get_device_properties(0).total_memory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class ResUNET(nn.Module):
#     def __init__(self, img_size):
#         super().__init__()
#         self.img_size=img_size 
#         self.loops = int(math.log2(img_size)-math.log2(64))
#         self.encoder = self.build_encoder()
#         self.decoder = self.build_decoder()
#         self.decoder_residual_blocks_list = []
#         self.encoder_residual_blocks_list = []
#         self.conv_blocks_list = []
        
#         self.build_encoder_residual_blocks()
#         self.build_decocer_residual_blocks()
#         self.build_conv_blocks()
#         self.build_intermediate_blocks()

        
#     def encoder_residualBlock(self,idx,x0):
#       c1,c2,r1,r2 = self.encoder_residual_blocks_list[idx]
#       # c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#       # c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       # r1 = ReLU()
#       # r2 = ReLU()
#       x = c1(x0)
#       x = r1(x)
#       x = c2(x)
#       x = x + x0
#       return r2(x)

#     def decoder_residualBlock(self,idx,x0):
#       c1,c2,c3,r1,r2,skip,upsample = self.decoder_residual_blocks_list[idx]  
#       # r1 = ReLU()
#       # r2 = ReLU()
#       # c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#       # c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       # c3 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       # skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
#       # upsample = Upsample(scale_factor=2,mode="bilinear",align_corners=True)
#       x = c1(x0)
#       x = r1(x)
#       x = c2(x)
#       x = skip(x0) + x
#       x = r2(x)
#       x = upsample(x)
#       x = c3(x)
#       return x

#     def convBlock(self,idx,x0):
#         c1,b1,r1 = self.conv_blocks_list[idx]
#         x2 = c1(x0)
#         x2 = b1(x2)
#         return r1(x2)


#     def build_encoder(self):
#       # Almacenamos los bloques de convoluciones, el bloque residual no lo incluimos
#       base = 2
#       res = []
#       for i in range(0,self.loops):
#         if i ==0:
#           res.append(Sequential(*[Conv2d(3,base**(6+i),kernel_size=3,padding=1),
#                                   BatchNorm2d(base**(6+i)),
#                                   ReLU(),
#                                   Conv2d(base**(6+i),base**(6+i),kernel_size=3,padding=1),
#                                   BatchNorm2d(base**(6+i)),
#                                   ReLU()                                  
#                                   ]))
#         else:
#           res.append(Sequential(*[Conv2d(base**(6+i-1),base**(6+i),kernel_size=3,padding=1),
#                                   BatchNorm2d(base**(6+i)),
#                                   ReLU(),
#                                   Conv2d(base**(6+i),base**(6+i),kernel_size=3,padding=1),
#                                   BatchNorm2d(base**(6+i)),
#                                   ReLU()
#                                   ]))
#       return res

#     def build_decoder(self):
#       res = []
#       for i in range(self.loops-1):
#         res.append(Sequential(*[Conv2d(self.img_size//(2**i),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
#                                 BatchNorm2d(self.img_size//(2**(i+1))),
#                                 ReLU(),
#                                 Conv2d(self.img_size//(2**(i+1)),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
#                                 BatchNorm2d(self.img_size//(2**(i+1))),
#                                 ReLU()
#                                 ]))
#       return res
    
#     def build_encoder_residual_blocks(self):
#       base = 2
#       for idx,layer in enumerate(self.encoder):
#         input_channels = base**(6+idx)
#         mid_channels = base**(6+idx)
#         c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#         c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#         r1 = ReLU()
#         r2 = ReLU()
#         self.encoder_residual_blocks_list.append([c1,c2,r1,r2])
    
#     def build_decocer_residual_blocks(self):
#       base = 2
#       for idx,layer in enumerate(self.encoder):
#         input_channels = self.img_size//(2**(idx+1))
#         mid_channels = self.img_size//(2**(idx+2))
#         r1 = ReLU()
#         r2 = ReLU()
#         c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#         c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#         c3 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#         skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
#         upsample = Upsample(scale_factor=2,mode="bilinear",align_corners=True)
#         self.decoder_residual_blocks_list.append([c1,c2,c3,r1,r2,skip,upsample])
        
#     def build_conv_blocks(self):
#       call_values = [
#         (self.img_size//2,self.img_size),\
#         (self.img_size,self.img_size),\
#         (self.img_size,self.img_size*2),\
#         (self.img_size*2,self.img_size*2),\
#         (self.img_size*2,self.img_size),
#         (self.img_size,self.img_size),\
#         (128,64),\
#         (64,1)
#         ]
#       for input_channel,mid_channels in call_values:
#         c1 = Conv2d(input_channel,mid_channels,kernel_size=3,padding=1)
#         b1 = BatchNorm2d(mid_channels)
#         r1 = ReLU()
#         self.conv_blocks_list.append([c1,b1,r1])
  
#     def build_intermediate_blocks(self):
#       input_channels = self.img_size
#       mid_channels = self.img_size
#       c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#       c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       r1 = ReLU()
#       r2 = ReLU()
      
#       self.encoder_residual_blocks_list.append([c1,c2,r1,r2])
      
#       input_channels = self.img_size*2
#       mid_channels = self.img_size
      
#       r1 = ReLU()
#       r2 = ReLU()
#       c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#       c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       c3 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
#       upsample = Upsample(scale_factor=2,mode="bilinear",align_corners=True)
      
#       self.decoder_residual_blocks_list.append([c1,c2,c3,r1,r2,skip,upsample])
      
#       input_channels = self.img_size
#       mid_channels = self.img_size//2
      
#       r1 = ReLU()
#       r2 = ReLU()
#       c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
#       c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       c3 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
#       skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
#       upsample = Upsample(scale_factor=2,mode="bilinear",align_corners=True)
      
#       self.decoder_residual_blocks_list.append([c1,c2,c3,r1,r2,skip,upsample])
      
        
            
#     def forward(self,x):
#         base = 2
#         # Empezamos haciendo la bajada de la U
#         x = self.encoder[0](x)
#         x = self.encoder_residualBlock(0,x)
#         x1 = MaxPool2d(2)(x)
#         # (feature sin maxpool, feature despues del maxpool)
#         features_encoder = [(x,x1)]
#         for idx,layer in enumerate(self.encoder[1:]):
#             x = features_encoder[-1][1]
#             x = layer(x)
#             # x = self.encoder_residualBlock(base**(6+idx+1),base**(6+idx+1),x)
#             x = self.encoder_residualBlock(idx+1,x)
#             x1 = MaxPool2d(2)(x)
#             features_encoder.append((x,x1))
            
        
#         # Intermedio

#         # x = self.convBlock(self.img_size//2,self.img_size,features_encoder[-1][1])
#         # x = self.convBlock(self.img_size,self.img_size,x)
        
#         x = self.convBlock(0,features_encoder[-1][1])
#         x = self.convBlock(1,x)

#         # x = self.encoder_residualBlock(self.img_size,self.img_size,x)
#         x = self.encoder_residualBlock(len(self.encoder_residual_blocks_list)-1,x)

#         x1 = MaxPool2d(2)(x)

#         # x1 = self.convBlock(self.img_size,self.img_size*2,x1)
#         # x1 = self.convBlock(self.img_size*2,self.img_size*2,x1)
        
#         x1 = self.convBlock(2,x1)
#         x1 = self.convBlock(3,x1)
        

#         # x1 = self.decoder_residualBlock(self.img_size*2,self.img_size,x1)
#         x1 = self.decoder_residualBlock(len(self.decoder_residual_blocks_list)-2,x1)
        

#         x = torch.cat([x,x1],dim=1)
        

#         # x = self.convBlock(self.img_size*2,self.img_size,x)
#         # x = self.convBlock(self.img_size,self.img_size,x)
        
#         x = self.convBlock(4,x)
#         x = self.convBlock(5,x)

#         # x = self.decoder_residualBlock(self.img_size,self.img_size//2,x)
#         x = self.decoder_residualBlock(len(self.decoder_residual_blocks_list)-1,x)
        

#         # Decoder
#         for idx,layer in enumerate(self.decoder):
#           x1 = features_encoder[-1-idx][0]
#           x = torch.cat([x1,x],dim=1)
#           x = layer(x)
#           # x = self.decoder_residualBlock(self.img_size//(2**(idx+1)),self.img_size//(2**(idx+2)),x)
#           x = self.decoder_residualBlock(idx,x)

#         x1 = features_encoder[0][0]
#         x = torch.cat([x1,x],dim=1)
        
#         # x = self.convBlock(128,64,x)
#         # x = self.convBlock(64,1,x)
        
#         x = self.convBlock(6,x)
#         x = self.convBlock(7,x)

#         return x

class ConvBlock(nn.Module):
  def __init__(self, input_channel, mid_channels):
    super().__init__()
    self.input_channel = input_channel
    self.mid_channels = mid_channels
    self.c1 = Conv2d(input_channel,mid_channels,kernel_size=3,padding=1)
    self.b1 = BatchNorm2d(mid_channels)
    self.r1 = ReLU()
  
  def forward(self,x):
        x = self.c1(x)
        x = self.b1(x)
        return self.r1(x)
    

class Encoder_residual_block(nn.Module):
    def __init__(self, input_channels, mid_channels):
      super().__init__()
      self.input_channels = input_channels
      self.c1 = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
      self.c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
      self.r1 = ReLU()
      self.r2 = ReLU()
    
    def forward(self,x0):
      x = self.c1(x0)
      x = self.r1(x)
      x = self.c2(x)
      x0 = x + x0
      del x
      return self.r2(x0)
      
class Encoder(nn.Module):
    def __init__(self, img_size, loops):
      super().__init__()
      self.img_size = img_size
      self.loops = loops
      self.encoder = self.build_encoder()
      self.residual_blocks = self.build_residual_blocks()
      self.features = []
    
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
                                  ]).to(device))
        else:
          res.append(Sequential(*[Conv2d(base**(6+i-1),base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU(),
                                  Conv2d(base**(6+i),base**(6+i),kernel_size=3,padding=1),
                                  BatchNorm2d(base**(6+i)),
                                  ReLU()
                                  ]).to(device))
      return res
    
    def build_residual_blocks(self):
      base = 2
      residual_blocks = []
      for idx,layer in enumerate(self.encoder):
        input_channels = base**(6+idx)
        mid_channels = base**(6+idx)
        residual_blocks.append(Encoder_residual_block(input_channels=input_channels,mid_channels=mid_channels).to(device))
        
      return residual_blocks      
    
    def forward(self,x):
        # Empezamos haciendo la bajada de la U
        x = self.encoder[0](x)
        x = self.residual_blocks[0](x)
        x1 = MaxPool2d(2)(x)
        # (feature sin maxpool, feature despues del maxpool)
        self.features = [[x,x1]]
        for idx,layer in enumerate(self.encoder[1:]):
            x = self.features[-1][1]
            x = layer(x)
            # x = self.encoder_residualBlock(base**(6+idx+1),base**(6+idx+1),x)
            x = self.residual_blocks[idx+1](x)
            x1 = MaxPool2d(2)(x)
            self.features.append([x,x1])
        
        del x1
        
        return x
      
class Intermediate(nn.Module):
  def __init__(self,img_size):
    super().__init__()
    self.img_size = img_size
    self.conv_block_1 = ConvBlock(self.img_size//2,self.img_size)
    self.conv_block_2 = ConvBlock(self.img_size,self.img_size)
    self.conv_block_3 = ConvBlock(self.img_size,self.img_size*2)
    self.conv_block_4 = ConvBlock(self.img_size*2,self.img_size*2)
    self.conv_block_5 = ConvBlock(self.img_size*2,self.img_size)
    self.conv_block_6 = ConvBlock(self.img_size,self.img_size)
    
    self.encoder_residual_block = Encoder_residual_block(self.img_size,self.img_size)
    self.decoder_residual_block_1 = Decoder_residual_block(self.img_size*2,self.img_size)
    self.decoder_residual_block_2 = Decoder_residual_block(self.img_size,self.img_size//2)
    
  
  def forward(self,x,features_encoder):
    
        # x = self.convBlock(self.img_size//2,self.img_size,features_encoder[-1][1])
        # x = self.convBlock(self.img_size,self.img_size,x)
        
        x = self.conv_block_1(features_encoder[-1][1])
        x = self.conv_block_2(x)

        # x = self.encoder_residualBlock(self.img_size,self.img_size,x)
        x = self.encoder_residual_block(x)

        x1 = MaxPool2d(2)(x)

        # x1 = self.convBlock(self.img_size,self.img_size*2,x1)
        # x1 = self.convBlock(self.img_size*2,self.img_size*2,x1)
        
        x1 = self.conv_block_3(x1)
        x1 = self.conv_block_4(x1)
        

        # x1 = self.decoder_residualBlock(self.img_size*2,self.img_size,x1)
        x1 = self.decoder_residual_block_1(x1)
        
        x = torch.cat([x,x1],dim=1)
        del x1

        # x = self.convBlock(self.img_size*2,self.img_size,x)
        # x = self.convBlock(self.img_size,self.img_size,x)
        
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)

        # x = self.decoder_residualBlock(self.img_size,self.img_size//2,x)
        x = self.decoder_residual_block_2(x)

        return x
    
            
class Decoder_residual_block(nn.Module):
  def __init__(self, input_channels, mid_channels):
      super().__init__()
      self.r1 = ReLU()
      self.r2 = ReLU()
      self.c1  = Conv2d(input_channels,mid_channels,kernel_size=3,padding=1)
      self.c2 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
      self.c3 = Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1)
      self.skip = Conv2d(input_channels,mid_channels,kernel_size=1,padding=0)
      self.upsample = Upsample(scale_factor=2,mode="bilinear",align_corners=True)
    
  def forward(self,x0):
      x = self.c1(x0)
      x = self.r1(x)
      x = self.c2(x)
      x = self.skip(x0) + x
      x = self.r2(x)
      x = self.upsample(x)
      x0 = self.c3(x)
      del x
      return x0
            
class Decoder(nn.Module):            
  def __init__(self, img_size, loops):
    super().__init__()
    self.img_size = img_size
    self.loops = loops
    self.decoder = self.build_decoder()
    self.residual_blocks = self.build_residual_blocks()
    
    self.conv_block_1 = ConvBlock(input_channel=128,mid_channels=64)
    self.conv_block_2 = ConvBlock(input_channel=64,mid_channels=1)
  
  def build_decoder(self):
    res = []
    for i in range(self.loops-1):
      res.append(Sequential(*[Conv2d(self.img_size//(2**i),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
                              BatchNorm2d(self.img_size//(2**(i+1))),
                              ReLU(),
                              Conv2d(self.img_size//(2**(i+1)),self.img_size//(2**(i+1)),kernel_size=3,padding=1),
                              BatchNorm2d(self.img_size//(2**(i+1))),
                              ReLU()
                              ]).to(device))
    return res
  
  def build_residual_blocks(self):
    residual_blocks = []
    for idx,layer in enumerate(self.decoder):
      residual_blocks.append(Decoder_residual_block(self.img_size//(2**(idx+1)),self.img_size//(2**(idx+2))).to(device))
      
    return residual_blocks
    
  
  def forward(self,x,encoder_features):
    for idx,layer in enumerate(self.decoder):
      x1 = encoder_features[-1-idx][0]
      x = torch.cat([x1,x],dim=1)
      x = layer(x)
      # x = self.decoder_residualBlock(self.img_size//(2**(idx+1)),self.img_size//(2**(idx+2)),x)
      x = self.residual_blocks[idx](x)

    x1 = encoder_features[0][0]
    x = torch.cat([x1,x],dim=1)
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    del x1
    return x
  
class ResUNET(nn.Module):
  def __init__(self, img_size):
    super().__init__()
    self.img_size = img_size 
    self.loops = int(math.log2(img_size)-math.log2(64))
    self.encoder = Encoder(img_size,self.loops)
    self.decoder = Decoder(img_size,self.loops)
    self.intermediate = Intermediate(img_size)
    
  def forward(self,x):
    x = self.encoder(x)
    x = self.intermediate(x,self.encoder.features)
    x = self.decoder(x,self.encoder.features)
    del self.encoder.features
    torch.cuda.empty_cache()
    gc.collect()
    return x
    
# define the LightningModule
class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = ResUNET2(img_size=self.hparams.img_size)
        # self.model.encoder.features = self.model.encoder.features.to(self.device)
        self.accuracy = Accuracy(task="binary")
        self.auc =  BinaryAUROC()
        self.recall = BinaryRecall(average="macro") 
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
      x, y = batch
      # x = x.to(self.device)
      # y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
      
      self.log("train_loss_step", loss,prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_acc_step", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_auroc_step", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      self.log("train_recall_step",self.recall(torch.round(yhat* torch.pow(10, torch.tensor(2))) / torch.pow(10, torch.tensor(2)),y),on_epoch=True,on_step=False)
      # self.log("train_recall_step",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
    
      return loss
    
    def training_epoch_end(self, training_step_outputs):
        self.log("step",self.current_epoch)
        
        
    def validation_step(self, batch, batch_idx):
      x,y = batch
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
        
      self.log("train_val_loss", loss,prog_bar=True)
      self.log("train_val_acc", self.accuracy(logits, y),prog_bar=True)
      self.log("train_val_auroc", self.auc(yhat, y),prog_bar=True)
      self.log("train_val_recall",self.recall(torch.round(yhat* torch.pow(10, torch.tensor(2))) / torch.pow(10, torch.tensor(2)),y))

      return loss
  
    def validation_epoch_end(self, validation_step_outputs):
      self.log("step",self.current_epoch)
  
      
    def test_step(self,batch,batch_idx):
      x,y = batch
      # x = x.to(self.device)
      # y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y.float())
      self.log("train_test_loss", loss)
      self.log("train_test_acc", self.accuracy(logits, y))
      self.log("train_test_auroc", self.auc(yhat, y))
      self.log("train_test_recall",self.recall(torch.argmax(yhat,dim=1),y))

      return loss
      
    def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
      return optimizer
  
# if __name__ == '__main__':
#     img_size = 1024
#     model = ResUNET(img_size)
#     with torch.no_grad():
#           model(torch.rand(1,3,img_size,img_size))

if __name__ == '__main__':
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
        "img_size":256,
        "num_workers":4,
        "lr":1e-3,
    }
    
    
    pl.seed_everything(42,workers=True)
    # train_features = UNETDataset("../train_unet/Unet_train.csv","../../train_unet/",transform=CustomTransformationResUnet(config["img_size"]))
    # train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    # val_dataset = UNETDataset("../train_unet/Unet_val.csv","../../train_unet/",transform=CustomTransformationResUnet(config["img_size"]))
    # val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)
   
    train_features = UNETDataset("./Unet_train.csv",CustomTransformationResUnet(config["img_size"]))
    train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    val_dataset = UNETDataset("./Unet_val.csv",CustomTransformationResUnet(config["img_size"]))
    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)

    # Initialize a trainer
    trainer = Trainer(
        accumulate_grad_batches=32, # acumula los gradientes de los primeros 4 batches
        #deterministic=True,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=1000,
        callbacks=[TQDMProgressBar(),
                   EarlyStopping(monitor="train_val_loss",mode="min",patience=3),
                   ModelCheckpoint(dirpath="./model-checkpoint-resUNET/",\
                    filename="resunet-{epoch}-{train_val_acc:.2f}",
                    save_top_k=2,
                    monitor="train_val_loss")],
        log_every_n_steps=40,
        # limit_train_batches=1.0, limit_val_batches=1.0
        # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt"
    )
    #test_loader = train_loader

    miopia_model = SegmentationModel(config)
    trainer.fit(miopia_model, train_loader,val_loader)