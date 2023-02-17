import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

class CustomTransformations():
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        orSize = (img.shape[1],img.shape[2])
        return transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(size=orSize,scale=(0.4,0.9)),
                transforms.Compose([
                transforms.RandomResizedCrop(size=orSize[0]//1.2,scale=(0.4,0.9)),
                transforms.Pad(int(self.size-(self.size//1.2)),fill=255)
                ])
            ]),
            transforms.Resize((self.size,self.size))
        ])(img)