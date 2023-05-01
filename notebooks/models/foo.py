# from resunet import ResUNET2, device
# import torch
  
# if __name__ == '__main__':
#     img_size = 1024
#     model = ResUNET2(img_size).to(device)
#     print(device)
#     with torch.no_grad():
#           model(torch.rand(1,3,img_size,img_size).to(device))
          
          
from vgg19 import VGG19TF
import torch
  
if __name__ == '__main__':
    img_size = 224
    model = VGG19TF(img_size)
    with torch.no_grad():
          model(torch.rand(4,3,img_size,img_size))