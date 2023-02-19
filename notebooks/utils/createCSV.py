from glob import glob 
import pandas as pd
import os
import json

def createCSV(dir_path,outputPath,json_path):
    df = pd.DataFrame(columns=["index","imgPath","label"])
    
    # Cargar archivo JSON
    with open(json_path) as json_file:
        data = json.load(json_file)
    json_photos = data['images']
    
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        print(img_path)
        if os.path.basename(img_path) in json_photos:
            print("Imagen omitida:", img_path)
            continue
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
    
if __name__ == '__main__':
    createCSV("../../test","../output.csv")