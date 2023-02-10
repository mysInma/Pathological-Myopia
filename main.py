import torch
import torchvision
import torchvision.transforms as transforms
import os 
import matplotlib.pyplot as plt
import numpy as np



TRAIN_PATH = './data/Training/PALM-Training400'

train_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)
                                       
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images,_ in loader:
        image_count_in_batch = images.size(0)
        print(images.shape)
        images = images.view(image_count_in_batch, images.size(1), -1)
        print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_batch
        
    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

print(get_mean_and_std(train_loader))
        
        
        

# mean=[0.485, 0.456, 0.406]; # Mean of the ImageNet dataset
# std=[0.229, 0.224, 0.225]; # Standard deviation of the ImageNet dataset

# train_transforms = transforms.Compose([
#     transform.Resize((224,224)), # Resize the image to 224x224 pixels
#     transform.RandomHorizontalFlip(), # Horizontally flip the image
#     transform.RandomRotation(10), # Rotate the image by 10 degrees
#     transform.ToTensor(), # Convert the image to a PyTorch Tensor
#     transform.Normalize(torch.Tensor(mean), torch.Tensor(std)) # Normalize the image
    