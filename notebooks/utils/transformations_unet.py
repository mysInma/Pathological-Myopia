from tqdm import tqdm
import cv2
from albumentations import HorizontalFlip, VerticalFlip, Rotate

class CustomTransformationsUnet():
    def __init__(self, size):
        self.size = size
        
    def __call__(self, images, masks, augment=True):
        self.size = (512,512)
        
        for x, y in tqdm(zip(images, masks), total=len(images)):
            
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            y = cv2.imread(y, cv2.IMREAD_COLOR)
            
            if augment == True:
                aug = HorizontalFlip(p=1.0)
                augmented = aug(image=x, mask=y)
                x1 = augmented["image"]
                y1 = augmented["mask"]
                
                aug = VerticalFlip(p=1.0)
                augmented = aug(image=x, mask=y)
                x2 = augmented["image"]
                y2 = augmented["mask"]
                
                aug = Rotate(limit=45, p=1.0)
                augmented = aug(image=x, mask=y)
                x3 = augmented["image"]
                y3 = augmented["mask"]
                
                
                X = [x, x1, x2, x3]
                Y = [y, y1, y2, y3]

                
if __name__ == '__main__':
    
    pass
    #path = ""
    #images, masks = load_data(path)     
    #CustomTransformationsUnet(images, masks, augment=True)
    
    
            