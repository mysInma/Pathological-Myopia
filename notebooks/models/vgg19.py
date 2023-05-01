import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19TF(nn.Module):
    def __init__(self, img_size, num_classes=3):
        super(VGG19TF, self).__init__()
        
        self.vgg19 = vgg19(weights=VGG19_Weights.IMAGENET1K_V1);
        
        #x_image = tf.reshape(self.x, [-1, self.pic_size, self.pic_size, 3])
        
        # Añadir la capa Conv1 y la capa Unpool2
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Añadir las capas FC adicionales
        self.fc1 = nn.Linear(128, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(128, num_classes)
        
    # def load_image(image_path, max_size=400, shape=None):
    #     image = Image.open(image_path).convert("RGB")
    #     if max(image.size)>max_size:
    #         size=max_size
    #     else:
    #         size = max(image.size)
        
    #     if shape is not None:
    #         size=shape
        
    #     tfms = transforms.Compose([
    #         transforms.Resize(size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            
    #     ])
        
    #     image = tfms(image).unsqueeze(0)
    #     return image
       

    def forward(self, x):
        
        #Pasar el modelo
        model = self.vgg19(x)
        
         # Extraer las dos últimas estaciones de VGG19
        # a1 = nn.Sequential(*list(model.features.children())[:-1])
        # b2 = nn.Sequential(*list(model.features.children())[:-2])
        
        # Acceder a las dos primeras capas convolucionales del cuarto bloque
        conv4_1 = model.features[34]
        conv4_2 = model.features[36]

        # Acceder a las dos primeras capas convolucionales del quinto bloque
        conv5_1 = model.features[44]
        conv5_2 = model.features[46]
        
        
        # Aplicar la convolución y el upsampling
        # x1 = self.avgpool(a)
        # x2 = self.conv1(b)
        # x3 = self.conv3(c)
        # x4 = self.unpool2(d)

        # Sumar x1 y x2
        x5 = x1 + x2
        
        # Sumar x3 y x4
        x6 = x3 + x4
        
        # Pasar a través de las capas FC
        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc1(x5)
        
        x6 = x6.view(x6.size(0), -1)
        x6 = self.fc2(x6)
        
        # Sumar x5 y x6
        x7 = x5 + x6
        
        # Pasar a través de la capa final FC
        x7 = self.fc3(x7)
        
        return x7
    
    
if __name__ == '__main__':
    model = VGG19TF(224)
    with torch.no_grad():
        for i in range(5):
          model(torch.rand(1,3,224,224))