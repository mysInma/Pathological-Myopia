from glob import glob 
import pandas as pd
import os

def createCSV(dir_path,outputPath):
    df = pd.DataFrame(columns=["index","imgPath","label"])
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        file_name = os.path.basename(img_path) #para extraer el nombre de archivo de img_path.
        label = encodingLabel(file_name.split("_")[0][0])
        df.loc[index] = [index,img_path,label]
    df.to_csv(outputPath,index=False)
def encodingLabel(label):
    if "H" in label:
        return 0
    elif "N" in label:
        return 1
    else:
        return 2