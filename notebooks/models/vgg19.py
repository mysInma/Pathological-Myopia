import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19TF(nn.Module):
    def __init__(self, img_size, num_classes=3):
        super(VGG19TF, self).__init__()
        # Extraer las dos últimas estaciones de VGG19
        features = nn.Sequential(*list( vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.children())[:-2])
        
        # Primera cabeza
        self.head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Segunda cabeza
        self.head2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Capa completamente conectada adicional para procesar la suma de las salidas de las dos cabezas
        self.fc = nn.Linear(256, num_classes)
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            self.fc
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        head1_output = self.head1(x)
        head2_output = self.head2(x)
        x = torch.cat([head1_output, head2_output], dim=1)
        x = self.fc(x)
        return x
