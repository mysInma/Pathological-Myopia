from glob import glob 
import pandas as pd
import os

def createCSV(dir_path,outputPath):
    df = pd.DataFrame(columns=["index","imgPath"])
    for index, img_path in enumerate(glob(os.path.join(dir_path,"*.jpg"))):
        df.loc[index] = [index,img_path]
    df.to_csv(outputPath,index=False)