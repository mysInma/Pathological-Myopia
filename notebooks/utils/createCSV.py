from glob import glob 
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
import numpy as np 

def createCSV(dir_path,outputPath,json_path):
    df = pd.DataFrame(columns=["imgPath","label"])
    
    # Cargar archivo JSON
    with open(json_path) as json_file:
        data = json.load(json_file)
    json_photos = data['images']
    
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        if os.path.basename(img_path) in json_photos:
            continue
        file_name = os.path.basename(img_path) #para extraer el nombre de archivo de img_path.
        label = encodingLabel(file_name.split("_")[0][0])
        df.loc[index] = [img_path,label]
    X_train, X_val, y_train, y_val = train_test_split(df["imgPath"], df["label"], test_size=0.2, random_state=42)
    pd.DataFrame({"imgPath": X_train.values, "label":y_train.values}).to_csv(os.path.join(outputPath,"train.csv"),index=False)
    pd.DataFrame({"imgPath": X_val.values, "label":y_val.values}).to_csv(os.path.join(outputPath,"val.csv"),index=False)
def encodingLabel(label):
    if "H" in label:
        return 0
    elif "N" in label:
        return 1
    else:
        return 2
    
if __name__ == '__main__':
    createCSV("../../test","../output.csv")