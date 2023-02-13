import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

class CustomTransformations(Compose):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transformations = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop((224,224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
    def __call__(self, tensor):
        return self.transformations(tensor)