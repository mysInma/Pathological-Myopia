import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

class CustomTransformations(self):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transformations = transforms.Compose([
             transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img):
        return self.transform(img)