from glob import glob 
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
import numpy as np 


def resnetCSV(dir_path: str,outputPath:str,json_path:str):
    """Creates two csvs in order to use a pytorch dataloader for a clasification model training. One csv is train.csv for training model and other is val.csv for validation.

    Args:
        dir_path (str): Path where the data is contained
        outputPath (str): Output path to save the resulting csv
        json_path (str): Json file path if you want to miss some data to the final csv
        
    JSON FORMAT:
                {
                    "images":[ "imageName1.jpg"]
                }
    """
    df = pd.DataFrame(columns=["imgPath","label"])
    
    # Cargar archivo JSON
    json_photos = readJSON(json_path)
    
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        if os.path.basename(img_path) in json_photos:
            continue
        file_name = os.path.basename(img_path) #para extraer el nombre de archivo de img_path.
        label = encodingLabel(file_name.split("_")[0][0])
        df.loc[index] = [img_path,label]
    X_train, X_val, y_train, y_val = train_test_split(df["imgPath"], df["label"], test_size=0.2, random_state=42)
    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        
    createTrainValCSV(X_train,X_val,y_train,y_val,"resnet","imgPath","label",outputPath)

def unetCSV(dir_path:str, mask_path:str, json_path:str, output_path:str):
    """_summary_

    Args:
        dir_path (str): _description_
        mask_path (str): _description_
        json_path (str): _description_
        output_path (str): _description_

    """
    
    if not os.path.exists(os.path.exists(dir_path)):
        raise Exception(f"{dir_path} is not a correct path")
    
    if not os.path.exists(os.path.exists(mask_path)):
        raise Exception(f"{mask_path} is not a correct path")
    
    if json_path and  (not os.path.exists(os.path.isfile(json_path))):
        raise Exception(f"{json_path} is not a correct path")
    
    df = pd.DataFrame(columns=["imgPath","maskPath"])
    
    json_photos = readJSON(json_path)
    
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        if os.path.basename(img_path) in json_photos:
            continue
        file_name = os.path.basename(img_path) #para extraer el nombre de archivo de img_path.
        mask_file = os.path.join(mask_path,file_name.split(".")[0]+".bmp")
        
        if not os.path.isfile(mask_file):
            continue
        
        df.loc[index] = [img_path,mask_file]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    X_train, X_val, y_train, y_val = train_test_split(df["imgPath"], df["maskPath"], test_size=0.2, random_state=42)
    
    createTrainValCSV(X_train,X_val,y_train,y_val,"Unet","imgPath","maskPath",output_path)
    
def readJSON(json_path:str):
    try:
        with open(json_path) as json_file:
            data = json.load(json_file)
        json_photos = data['images']
    except:
        json_photos = []
    return json_photos

def createTrainValCSV(X_train:pd.Series, X_val:pd.Series, y_train:pd.Series, y_val:pd.Series,csv_name:str,x_name:str,y_name:str,output_path:str):
    """Creates two csv's one for training and the other for validation

    Args:
        X_train (pd.Series): imgPath split for training
        X_val (pd.Series): imgPath split for validation
        y_train (pd.Series): data tarjet path for trainint
        y_val (pd.Series): data tarjet path for validation
        csv_name (str): name of css
        x_name (str): label name in csv for X_train/val
        y_name (str): label name in csv for y_train/val
        output_path (str): path where the csv's are going to be saved
    """
    pd.DataFrame({x_name: X_train.values, y_name:y_train.values}).to_csv(os.path.join(output_path,f"{csv_name}_train.csv"),index=False)
    
    pd.DataFrame({x_name: X_val.values, y_name:y_val.values}).to_csv(os.path.join(output_path,f"{csv_name}_val.csv"),index=False)
    
    
def encodingLabel(label):
    if "N" in label:
        return 0
    elif "P" in label:
        return 1
    else:
        return 2
    
#if __name__ == '__main__':
    #createCSV("../../test","../output.csv")
