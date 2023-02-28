from unittest import mock
import sys
sys.path.insert(1,"/workspace/fast-api")
import os
from app.main import app
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

def test_read_main():
    # print()
    # print(os.getcwd())
    # print("Hola")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
@pytest.mark.parametrize("filename", [("/workspace/fast-api/images/prueba.jpg")])
def test_read_image(filename):
    # print("mi alma")
    # print(filename)
    response = client.post(
    "/uploadfile", files={"file": (filename, open(filename, "rb"), "image/jpeg")})
    # print(response.json())
    assert response.status_code == 200
from io import BytesIO
from PIL import Image, ImageChops
from functools import reduce
@pytest.mark.parametrize("filename", [("/workspace/fast-api/images/prueba.jpg"),("/workspace/fast-api/images/prueba2.jpg")])
def test_image_send(filename):
    response = client.post(
    "/sendImage", files={"file": (filename, open(filename, "rb"), "image/jpeg")})
    print(filename)
    im = Image.open(BytesIO(response.content)).convert("RGB")   
    im2 = Image.open(filename).convert("RGB")
    diff = ImageChops.difference(im,im2)
    res = False
    if len(set(diff.getdata()))/reduce(lambda x,y: x*y,im.size)<10e-2 :
        res = True
    assert res == True
    