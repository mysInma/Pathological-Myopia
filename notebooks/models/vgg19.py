import torch
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import math
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.classification import Accuracy, BinaryAUROC, BinaryRecall
from torch import optim, nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint



class VGG19TF(nn.Module):
    def __init__(self, img_size):
        super(VGG19TF, self).__init__()
        
        self.vgg19 = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.baseInputLayers = self.init_input_layer(img_size)
        
        #Bloque 1 por arriba
        self.conv_11_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.avgpool_11_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_11_1 = nn.ReLU()
        
        self.conv_11_2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.avgpool_11_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_11_2 = nn.ReLU()
        
        self.conv_11_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.avgpool_11_3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_11_3 = nn.ReLU()
        
        self.conv_11_4 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.avgpool_11_4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_11_4 = nn.ReLU()
        
        #Bloque 2 por arriba
        self.conv_21_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_21_1 = nn.ReLU()
        
        self.conv_21_2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_21_2 = nn.ReLU()
        
        self.conv_21_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_21_3 = nn.ReLU()
        
        self.conv_21_4 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_21_4 = nn.ReLU()
        
        #Bloque 1 por abajo
        self.conv_12_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_12_1 = nn.ReLU()
        
        self.conv_12_2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_12_2 = nn.ReLU()
        
        self.conv_12_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_12_3 = nn.ReLU()
        
        self.conv_12_4 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_12_4 = nn.ReLU()
        
        
        #Bloque 2 por abajo
        self.conv_22_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.unsample_22_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu_22_1 = nn.ReLU()
        
        self.conv_22_2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.unsample_22_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu_22_2 = nn.ReLU()
        
        self.conv_22_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.unsample_22_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu_22_3 = nn.ReLU()
        
        self.conv_22_4 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.unsample_22_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu_22_4 = nn.ReLU()
        
        
        #Capa FC por arriba
        self.fc1_1 = nn.Linear(25088, 512)
        self.relu_fc1_1 = nn.ReLU()
        self.fc1_2 = nn.Linear(512, 512)
        self.relu_fc1_2 = nn.ReLU()
        self.fc1_3 = nn.Linear(512,2)
        self.relu_fc1_3 = nn.ReLU()
        
        #Capa FC por abajo
        self.fc2_1 = nn.Linear(100352, 512)
        self.relu_fc2_1 = nn.ReLU()
        self.fc2_2 = nn.Linear(512, 512)
        self.relu_fc2_2 = nn.ReLU()
        self.fc2_3 = nn.Linear(512,2)
        self.relu_fc2_3 = nn.ReLU()
        
        
    def init_input_layer(self,image_size):
        exp_img_size = math.log2(image_size)
        exp_img_size_dest = math.log2(224)
        arch_layers = []
        for _ in range(int(exp_img_size-exp_img_size_dest)):
            arch_layers.append(nn.Conv2d(3,3,(3,3),padding=1,stride=1))
            arch_layers.append(nn.ReLU())
        return nn.Sequential(*arch_layers)
    
    def forward(self, images):
        
        x = self.baseInputLayers(images)
        
        #Extraer capas
        return_nodes = {
            "features.20": "b1_20",
            "features.22": "b1_22",
            "features.24": "b1_24",
            "features.26": "b1_26",
            
            "features.29": "b1_29",
            "features.31": "b1_31",
            "features.33": "b1_33",
            "features.35": "b1_35",
        
    }
        model2 = create_feature_extractor(self.vgg19, return_nodes=return_nodes)
        intermediate_outputs = model2(x)
        
        
        #Bloque 1 por arriba
        conv_11_1 = self.conv_11_1(intermediate_outputs['b1_20'])
        avgpool_11_1 = self.avgpool_11_1(conv_11_1)
        relu_11_1 = self.relu_11_1(avgpool_11_1)
        
        conv_11_2 = self.conv_11_2(intermediate_outputs['b1_22'])
        avgpool_11_2 = self.avgpool_11_2(conv_11_2)
        relu_11_2 = self.relu_11_2(avgpool_11_2)
        
        conv_11_3 = self.conv_11_3(intermediate_outputs['b1_24'])
        avgpool_11_3 = self.avgpool_11_3(conv_11_3)
        relu_11_3 = self.relu_11_3(avgpool_11_3)
        
        conv_11_4 = self.conv_11_4(intermediate_outputs['b1_26'])
        avgpool_11_4 = self.avgpool_11_4(conv_11_4)
        relu_11_4 = self.relu_11_4(avgpool_11_4)
        
        
        #Bloque 2 por arriba
        conv_21_1 = self.conv_21_1(intermediate_outputs['b1_29'])
        relu_21_1 = self.relu_21_1(conv_21_1)
        
        conv_21_2 = self.conv_21_2(intermediate_outputs['b1_31'])
        relu_21_2 = self.relu_21_2(conv_21_2)
        
        conv_21_3 = self.conv_21_3(intermediate_outputs['b1_33'])
        relu_21_3 = self.relu_21_3(conv_21_3)
        
        conv_21_4 = self.conv_21_4(intermediate_outputs['b1_35'])
        relu_21_4 = self.relu_21_4(conv_21_4)
        
        
        sum1 = relu_11_1 + relu_11_2 + relu_11_3 + relu_11_4 + relu_21_1 + relu_21_2 + relu_21_3 + relu_21_4
        
        
         #Bloque 1 por abajo
        conv_12_1 = self.conv_12_1(intermediate_outputs['b1_20'])
        relu_12_1 = self.relu_12_1(conv_12_1)
        
        conv_12_2 = self.conv_12_2(intermediate_outputs['b1_22'])
        relu_12_2 = self.relu_12_2(conv_12_2)
        
        conv_12_3 = self.conv_12_3(intermediate_outputs['b1_24'])
        relu_12_3 = self.relu_12_3(conv_12_3)
        
        conv_12_4 = self.conv_12_4(intermediate_outputs['b1_26'])
        relu_12_4 = self.relu_12_4(conv_12_4)
        
        
        #Bloque 2 por abajo
        conv_22_1 = self.conv_22_1(intermediate_outputs['b1_29'])
        unsample_22_1 = self.unsample_22_1(conv_22_1)
        relu_22_1 = self.relu_22_1(unsample_22_1)
        
        conv_22_2 = self.conv_22_2(intermediate_outputs['b1_31'])
        unsample_22_2 = self.unsample_22_2(conv_22_2)
        relu_22_2 = self.relu_22_2(unsample_22_2)
        
        conv_22_3 = self.conv_22_3(intermediate_outputs['b1_33'])
        unsample_22_3 = self.unsample_22_3(conv_22_3)
        relu_22_3 = self.relu_22_3(unsample_22_3)
        
        conv_22_4 = self.conv_22_4(intermediate_outputs['b1_35'])
        unsample_22_4 = self.unsample_22_4(conv_22_4)
        relu_22_4 = self.relu_22_4(unsample_22_4)
        
   
        sum2 = relu_12_1 + relu_12_2 + relu_12_3 + relu_12_4 + relu_22_1 + relu_22_2 + relu_22_3 + relu_22_4 
        
        
        #En forma de vector unidimensional
        sum1_1 = sum1.flatten(start_dim=1)
        sum2_2 = sum2.flatten(start_dim=1)
        
        fc1_1 = self.fc1_1(sum1_1)
        relu_fc1_1 = self.relu_fc1_1(fc1_1)
        fc1_2 = self.fc1_2(relu_fc1_1)
        relu_fc1_2 = self.relu_fc1_2(fc1_2)
        fc1_3 = self.fc1_3(relu_fc1_2)
        relu_fc1_3 = self.relu_fc1_3(fc1_3)
        
        
        fc2_1 = self.fc2_1(sum2_2)
        relu_fc2_1 = self.relu_fc2_1(fc2_1)
        fc2_2 = self.relu_fc2_2(relu_fc2_1)
        relu_fc2_2 = self.relu_fc2_2(fc2_2)
        fc2_3 = self.fc2_3(relu_fc2_2)
        relu_fc2_3 = self.relu_fc2_3(fc2_3)

        
        output = relu_fc1_3 + relu_fc2_3
        
        return output
    
    
# define the LightningModule
class VGGModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = VGG19TF(img_size=self.hparams.img_size)
        self.accuracy = Accuracy(task="binary")
        self.auc = BinaryAUROC()
        self.recall = BinaryRecall(average="macro") 
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.pairwise_distance(yhat, y.float(), p=2).mean()
      
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
        print(batch)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        logits = self(x)
        yhat = torch.sigmoid(logits)
        loss = F.pairwise_distance(yhat, y.float(), p=2).mean()
        
        self.log("train_val_loss", loss,prog_bar=True)
        self.log("train_val_acc", self.accuracy(logits, y),prog_bar=True)
        self.log("train_val_auroc", self.auc(yhat, y),prog_bar=True)
        self.log("train_val_recall",self.recall(torch.round(yhat* torch.pow(10, torch.tensor(2))) / torch.pow(10, torch.tensor(2)),y))

        return loss
  
    def validation_epoch_end(self, validation_step_outputs):
      self.log("step",self.current_epoch)
  
      
    def test_step(self,batch,batch_idx):
      x,y = batch
      logits = self(x)
      yhat = torch.sigmoid(logits)
      loss = F.pairwise_distance(yhat, y.float(), p=2).mean()
    
      self.log("train_test_loss", loss)
      self.log("train_test_acc", self.accuracy(logits, y))
      self.log("train_test_auroc", self.auc(yhat, y))
      self.log("train_test_recall",self.recall(torch.argmax(yhat,dim=1),y))

      return loss
      
    def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
      return optimizer
        
    
    
if __name__ == '__main__':
    # model = VGG19TF(224)
    # with torch.no_grad():
    #     for i in range(5):
    #       model(torch.rand(4,3,224,224))
    
    import sys
    sys.path.insert(0,"../")
    from utils.dataloader import VGGDataset
    from utils.transformations import CustomTransformationVgg
    from models.vgg19 import VGG19TF
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from torchmetrics.functional import accuracy
    import torch
    from torch.utils.data import DataLoader
    
    config = {
        "batch_size":4,
        "img_size":224,
        "num_workers":4,
        "lr":1e-3,
    }
    
    
    pl.seed_everything(42,workers=True)
    
    # train_features = VGGDataset("../train_vgg/VGG_train.csv","../../train_vgg/",CustomTransformationVgg(config["img_size"]))
    # train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    # val_dataset = VGGDataset("../train_vgg/VGG_val.csv","../../train_vgg/",CustomTransformationVgg(config["img_size"]))
    # val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=False)
    
    train_features = VGGDataset("../train_vgg/VGG_train.csv","../../train_vgg/",None)
    train_loader = DataLoader(train_features,batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    
    val_dataset = VGGDataset("../train_vgg/VGG_val.csv","../../train_vgg/", None)
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
                   ModelCheckpoint(dirpath="./model-checkpoint-VGG19/",\
                    filename="vgg-{epoch}-{train_val_acc:.2f}",
                    save_top_k=2,
                    monitor="train_val_loss")],
        log_every_n_steps=40,
        # limit_train_batches=1.0, limit_val_batches=1.0
        # resume_from_checkpoint="some/path/to/my_checkpoint.ckpt"
    )
    #test_loader = train_loader

    miopia_model = VGGModel(config)
    trainer.fit(miopia_model, train_loader,val_loader)