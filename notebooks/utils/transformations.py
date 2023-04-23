import torch
from  torchvision import transforms
import cv2
from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensor

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

class CustomTransformationResUnet():
    def __init__(self, size):
        self.transformations = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Resize(size,size),HorizontalFlip(p=0.4),VerticalFlip(p=0.4),Rotate(p=0.2),ToTensor()])
        
    def __call__(self, image_path, mask_path):
        
        x = cv2.imread(image_path)
        y = cv2.imread(mask_path)
        
        transformed = self.transformations(image=x,mask=y)
        
        return transformed["image"],transformed["mask"]
