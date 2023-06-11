import pytorch_lightning as pl
from torch import Tensor
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import Accuracy   
from torch.nn import Conv2d, MaxPool2d, Upsample, ReLU, BatchNorm2d, Sequential
from torchmetrics.classification import BinaryRecall, BinaryAUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


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
    
    self.max_pool = MaxPool2d(2)
    
  
  def forward(self,x):
    
        # x = self.convBlock(self.img_size//2,self.img_size,features_encoder[-1][1])
        # x = self.convBlock(self.img_size,self.img_size,x)
        
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        # x = self.encoder_residualBlock(self.img_size,self.img_size,x)
        x = self.encoder_residual_block(x)

        x1 = self.max_pool(x)

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

class EfficientResUNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        # Encoder
        
        # lv1
        self.e1 = ConvBlock(3,64)
        self.e11 = ConvBlock(64,64)
        self.e1_res_block = Encoder_residual_block(64,64)
        self.e1_down = MaxPool2d(2)
        
        #lv2
        self.e2 = ConvBlock(64,128)
        self.e22 = ConvBlock(128,128)
        self.e2_res_block = Encoder_residual_block(128,128)
        self.e2_down = MaxPool2d(2)
        
        #lv3
        self.e3 = ConvBlock(128,256)
        self.e33 = ConvBlock(256,256)
        self.e3_res_block = Encoder_residual_block(256,256)
        self.e3_down = MaxPool2d(2)
        
        self.intermediate = Intermediate(512)
        
        # Decoder
        
        #lv3
        self.d3 = ConvBlock(512,256)
        self.d33 = ConvBlock(256,256)
        self.d3_res_block = Decoder_residual_block(256,128)
        
        #lv2
        self.d2 = ConvBlock(256,128)
        self.d22 = ConvBlock(128,128)
        self.d2_res_block = Decoder_residual_block(128,64)
        
        #lv1
        self.d1 = ConvBlock(128,64)
        self.d11 = ConvBlock(64,1)
        
    
    def forward(self,x):
        
        #Encoder
        
        #lv1
        x = self.e1(x)
        x = self.e11(x)
        x_f1 = self.e1_res_block(x)
        x = self.e1_down(x_f1)
        
        #lv2
        x = self.e2(x)
        x = self.e22(x)
        x_f2 = self.e2_res_block(x)
        x = self.e2_down(x_f2)
        
        #lv3
        x = self.e3(x)
        x = self.e33(x)
        x_f3 = self.e3_res_block(x)
        x = self.e3_down(x_f3)
        
        # Intermediate
        x = self.intermediate(x)
        
        # Decoder
        
        # lv3
        x = torch.cat([x,x_f3],dim=1)
        x = self.d3(x)
        x = self.d33(x)
        x = self.d3_res_block(x)
        
        # lv2
        x = torch.cat([x,x_f2],dim=1)
        x = self.d2(x)
        x = self.d22(x)
        x = self.d2_res_block(x)
        
        # lv1
        x = torch.cat([x,x_f1],dim=1)
        x = self.d1(x)
        x = self.d11(x)
        
        del x_f1,x_f2,x_f3
        
        return x

# if __name__ == '__main__':
#     img_size = 512
#     model = EfficientResUNET()
#     with torch.no_grad():
#           model(torch.rand(4,3,img_size,img_size))


class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = EfficientResUNET()
        # self.model.encoder.features = self.model.encoder.features.to(self.device)
        self.accuracy = Accuracy(task="binary")
        self.auc =  BinaryAUROC()
        self.criterion = nn.BCEWithLogitsLoss()
       # self.recall = BinaryRecall(average="macro") 
       
        self.train_loss_list = []
        self.train_acc_list = []
        self.train_auc_list = []
        
        self.avg_train_loss_list = []
        self.avg_train_acc_list = []
        self.avg_train_auroc_list = []
        
        
        
        self.val_loss_list = []
        self.val_acc_list = []
        self.val_auroc_list = []
          
        self.avg_val_loss_list = []
        self.avg_val_acc_list = []
        self.avg_val_auroc_list = []
          
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
    
      x, y = batch
      # x = x.to(self.device)
      # y = y.to(self.device)
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y)
      loss += self.dice_coeff(yhat.squeeze(1),y.squeeze(1))
      
      # self.log("train_loss_step", loss,prog_bar=True,on_epoch=True,on_step=False)
      # self.log("train_acc_step", self.accuracy(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      # self.log("train_auroc_step", self.auc(yhat, y),prog_bar=True,on_epoch=True,on_step=False)
      #self.log("train_recall_step",self.recall(torch.round(yhat* torch.pow(10, torch.tensor(2))) / torch.pow(10, torch.tensor(2)),y),on_epoch=True,on_step=False)
      # self.log("train_recall_step",self.recall(torch.argmax(yhat,dim=1),y),on_epoch=True,on_step=False)
      
      self.train_loss_list.append(loss.item())
      self.train_acc_list.append(self.accuracy(yhat, y).item())
      self.train_auc_list.append(self.auc(yhat, y).item())
      
      return loss
    
    def training_epoch_end(self, training_step_outputs):
      
      # Calcula la media de los tensores
      avg_loss = torch.tensor(self.train_loss_list).mean()
      avg_acc = torch.tensor(self.train_acc_list).mean()
      avg_auc = torch.tensor(self.train_auc_list).mean()

      self.train_loss_list.clear()
      self.train_acc_list.clear()
      self.train_auroc_list.clear()
      
      self.avg_train_loss_list.append(avg_loss.item())
      self.avg_train_acc_list.append(avg_acc.item())
      self.avg_train_auc_list.append(avg_auc.item())
    
    
      self.log("step",self.current_epoch)
        
        
    def validation_step(self, batch, batch_idx):
      
      x,y = batch
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.binary_cross_entropy(yhat, y)
      loss += self.dice_coeff(yhat.squeeze(1),y.squeeze(1))
        
      # self.log("train_val_loss", loss,prog_bar=True)
      # self.log("train_val_acc", self.accuracy(logits, y),prog_bar=True)
      # self.log("train_val_auroc", self.auc(yhat, y),prog_bar=True)
      #self.log("train_val_recall",self.recall(torch.round(yhat* torch.pow(10, torch.tensor(2))) / torch.pow(10, torch.tensor(2)),y))
        
      self.val_loss_list.append(loss.item())
      self.val_acc_list.append(self.accuracy(logits, y).item())
      self.val_auroc_list.append(self.auc(yhat, y).item())

      return loss
  
    def validation_epoch_end(self, validation_step_outputs):
      avg_val_loss = torch.tensor(self.val_loss_list).mean()
      avg_val_acc = torch.tensor(self.val_acc_list).mean()
      avg_val_auroc = torch.tensor(self.val_auroc_list).mean()

      self.avg_val_loss_list.append(avg_val_loss.item())
      self.avg_val_acc_list.append(avg_val_acc.item())
      self.avg_val_auroc_list.append(avg_val_auroc.item())

      self.val_loss_list.clear()
      self.val_acc_list.clear()
      self.val_auroc_list.clear()
      
      self.log("step",self.current_epoch)
      
    def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
      return optimizer
    
    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
      sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

      inter = 2 * (input * target).sum(dim=sum_dim)
      sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
      sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

      dice = (inter + epsilon) / (sets_sum + epsilon)
      return dice.mean()
  

if __name__ == '__main__':
    import sys
    sys.path.insert(0,"../")
    from utils.dataloader import UNETDataset
    from utils.transformations import CustomTransformationResUnet
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    import torch
    from torch.utils.data import DataLoader
    from utils.createCSV import unetCSV
    from pytorch_lightning import loggers as pl_loggers
    
    config = {
        "batch_size":2,
        "img_size":512,
        "num_workers":2,
        "lr":1e-3,
    }
    
    
    # unetCSV("../../data/PALM-Training400",
    #         "../../data/PALM-Training400-Annotation-D&F/Disc_Masks",
    #         "../utils/customDataVgg.json",
    #         "../datasets/efficient_resunet/")
    
    pl.seed_everything(42,workers=True)

   
    train_features = UNETDataset("../datasets/efficient_resunet/Unet_train.csv",CustomTransformationResUnet(config["img_size"]))
    train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    val_dataset = UNETDataset("../datasets/efficient_resunet/Unet_val.csv",CustomTransformationResUnet(config["img_size"]))
    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)

    # Initialize a trainer
    trainer = Trainer(
        accumulate_grad_batches=32, # acumula los gradientes de los primeros 4 batches
        #deterministic=True,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=1000,
        min_epochs=10,
        logger=pl_loggers.TensorBoardLogger("../logs/lightning_logs/efficient_resunet"),
        callbacks=[TQDMProgressBar(),
                   EarlyStopping(monitor="val_loss_list",mode="min",patience=3),
                    ModelCheckpoint(dirpath="../logs/model-checkpoints/model-checkpoint-resUNET/",\
                     filename="resunet-{epoch}-{train_val_acc:.2f}",
                     save_top_k=2,
                     monitor="val_loss_list")],
        
        log_every_n_steps=40,
        
        # limit_train_batches=1.0, limit_val_batches=1.0
        # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt"
    )
    #test_loader = train_loader

    miopia_model = SegmentationModel(config)
    trainer.fit(miopia_model, train_loader,val_loader)