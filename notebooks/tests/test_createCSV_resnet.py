import os
from pathlib import PosixPath
import sys
import pandas as pd
import pytest
from glob import glob
import json
sys.path.append(os.path.abspath(os.path.join('.', '.')))

from utils.createCSV import resnetCSV

tmp_path = PosixPath("PYTEST_TMPDIR/")

@pytest.mark.parametrize("dir_path,json_name", [("../test","jsonTest1.json"),("../test","jsonTest2.json")])
def test_create_csv_resnet(tmp_path,dir_path,json_name):
    tmp = tmp_path / "sub"
    tmp.mkdir()
    json_path = os.path.join(os.getcwd(),"tests/json",json_name)
    dir_path = os.path.join(os.getcwd(),dir_path)
    resnetCSV(dir_path,tmp,json_path)
    val = pd.read_csv(os.path.join(tmp,"resnet_val.csv"))
    train = pd.read_csv(os.path.join(tmp,"resnet_train.csv"))
    assert val.shape[0] != 0
    assert train.shape[0] != 0
    cwd = os.getcwd()
    os.chdir(tmp)
    
    # Files can be readed from tmp
    for _ ,row in train.iterrows():
        assert os.path.isfile(row["imgPath"]), f"file {os.path.basename(row['imgPath'])} cannot be accesed from {row['imgPath']}"
        
    for _ ,row in val.iterrows():
        assert os.path.isfile(row["imgPath"]), f"file {os.path.basename(row['imgPath'])} cannot be accesed from {row['imgPath']}"
        
    os.chdir(cwd)
    
    # check missed files
    
    df = pd.concat([train,val],ignore_index=True)
    
    images_readed = set(df["imgPath"].apply(lambda x: os.path.basename(x)).tolist())
    test_images = set([os.path.basename(x) for x in glob(os.path.join(dir_path,"*"))])
    
    with open(json_path) as json_file:
        data = json.load(json_file)
        miss_files = set(data['images'])
    
    assert len(images_readed.intersection(miss_files))==0, "There are not missed files"
    
    assert len(images_readed.intersection(test_images))==len(images_readed), "Not all files are in the csv"
    
@pytest.fixture(scope="session")
def csv_generation(tmp_path_factory):
    def _csv_generation(dir_path,json_name):
        fn = tmp_path_factory.mktemp("tmp") 
        json_path = os.path.join(os.getcwd(),"tests/json",json_name)
        resnetCSV(dir_path=dir_path,outputPath=fn,json_path=json_path)
        return fn
    return _csv_generation

@pytest.mark.parametrize("dir_path,json_name", [("../test","jsonTest1.json"),("../test","jsonTest2.json")])
def test_no_empty_csv_resnet(csv_generation,dir_path,json_name):
    path = csv_generation(dir_path,json_name)
    val = pd.read_csv(os.path.join(path,"resnet_val.csv"))
    train = pd.read_csv(os.path.join(path,"resnet_train.csv"))
    assert val.shape[0] != 0
    assert train.shape[0] != 0
    
@pytest.mark.parametrize("dir_path,json_name", [("../test","jsonTest1.json"),("../test","jsonTest2.json")])
def test_read_files_resnet(csv_generation,dir_path,json_name):
    path = csv_generation(dir_path,json_name)
    val = pd.read_csv(os.path.join(path,"resnet_val.csv"))
    train = pd.read_csv(os.path.join(path,"resnet_train.csv"))
    
    for _ ,row in train.iterrows():
        assert os.path.isfile(row["imgPath"]), f"file {os.path.basename(row['imgPath'])} cannot be accesed from {row['imgPath']}"
        
    for _ ,row in val.iterrows():
        assert os.path.isfile(row["imgPath"]), f"file {os.path.basename(row['imgPath'])} cannot be accesed from {row['imgPath']}"
    
@pytest.mark.parametrize("dir_path,json_name", [("../test","jsonTest1.json"),("../test","jsonTest2.json")])
def test_missed_files_resnet(csv_generation,dir_path,json_name):
    path = csv_generation(dir_path,json_name)
    val = pd.read_csv(os.path.join(path,"resnet_val.csv"))
    train = pd.read_csv(os.path.join(path,"resnet_train.csv"))
    df = pd.concat([train,val],ignore_index=True)
    json_path = os.path.join(os.getcwd(),"tests/json",json_name)
    
    images_readed = set(df["imgPath"].apply(lambda x: os.path.basename(x)).tolist())
    test_images = set([os.path.basename(x) for x in glob(os.path.join(dir_path,"*"))])
    
    with open(json_path) as json_file:
        data = json.load(json_file)
        miss_files = set(data['images'])
    
    assert len(images_readed.intersection(miss_files))==0, "There are not missed files"
    
    assert len(images_readed.intersection(test_images))==len(images_readed), "Not all files are in the csv"

@pytest.mark.parametrize("dir_path,json_name", [("../test","jsonTest3.json")])
def test_create_csv_bad_json_resnet(csv_generation,dir_path,json_name):
    path = csv_generation(dir_path,json_name)
    val = pd.read_csv(os.path.join(path,"resnet_val.csv"))
    train = pd.read_csv(os.path.join(path,"resnet_train.csv"))
    df = pd.concat([train,val],ignore_index=True)
    
    assert val.shape[0] != 0
    assert train.shape[0] != 0
    
    images_readed = len(df["imgPath"].apply(lambda x: os.path.basename(x)).tolist())
    test_images = len([os.path.basename(x) for x in glob(os.path.join(dir_path,"*.jpg"))])
    assert images_readed == test_images
