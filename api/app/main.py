from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from io import BytesIO
import onnxruntime as ort
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks
import os
import uuid

description = """
This API's porpuse is to make an interactive tool in order to use deep learning models. The models that can be tested are the following:

## Toy Model

A **super resolution** model to play and test with the API. This model is outside from the project scope.

- input: image of size (1,1,224,224)
- output: image of size (3,672,672)
"""

app = FastAPI(title="Deep Learning API",description=description)


@app.get("/")
async def root():
    return {"message": "Hello World"}

# ojo jpg no soporta RGBA pero png si
@app.post("/toy")
async def toy_model(file: UploadFile,background_tasks: BackgroundTasks ):
    content = await file.read()
    img = Image.open(BytesIO(content))
    final_img = inference_toy_model(img)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    id = uuid.uuid4()
    final_img.save(f"tmp/{id}.jpg")
    background_tasks.add_task(remove_file, f"tmp/{id}.jpg")
    return FileResponse(f"tmp/{id}.jpg")

def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/sendImage")
async def sendImage(file: UploadFile,background_tasks: BackgroundTasks):
    content = await file.read()
    im = Image.open(BytesIO(content)).convert("RGB")
    im.save("tmp/test.jpg")
    background_tasks.add_task(remove_file, "tmp/test.jpg")
    return FileResponse("tmp/test.jpg")


def inference_toy_model(img: Image.Image):
    if img.size != (224,224):
        img = img.resize((224,224))
    img_ycbcr = img.convert('YCbCr')
    ort_session = ort.InferenceSession('./models/super_resolution.onnx')
    img_y, img_cb, img_cr = img_ycbcr.split()
    input1 = np.expand_dims(np.expand_dims(np.array(img_y, dtype=np.float32),axis=0),axis=0)
    ort_inputs = {"input": input1}
    ort_output = ort_session.run(None, ort_inputs)[0]
    img_out_y = Image.fromarray(np.uint8((ort_output[0,0,:,:]).clip(0, 255)), mode='L')
    return Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

