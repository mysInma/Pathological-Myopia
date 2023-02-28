from unittest import mock
import sys
sys.path.insert(1,"/workspace/fast-api")
import os
from app.main import app, inference_toy_model
from fastapi.testclient import TestClient
import pytest
from PIL import Image
from io import BytesIO

client = TestClient(app)

@pytest.mark.parametrize("filename", [("images/cat12.jpg"),("images/prueba.jpg"),("images/prueba.jpg")])
def test_toy_inference(filename):
    img = Image.open(filename)
    img_infered = inference_toy_model(img)
    assert img_infered.size == (672,672)
    
@pytest.mark.parametrize("filename", [("images/prueba.jpg")])
def test_image_send(filename):
    response = client.post(
    "/sendImage", files={"file": (filename, open(filename, "rb"), "image/jpeg")})
    assert response.status_code == 200
    
    