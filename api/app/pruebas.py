from fastapi import FastAPI, Query # Query te permite hacer validaciones sobre los query parameters
from enum import Enum
from fastapi import File, UploadFile


description = """
ChimichangApp API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## Users

You will be able to:

* **Create users** (_not implemented_).
* **Read users** (_not implemented_).
"""

app = FastAPI(title="mis pruebas",description=description)


@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id:int):
#     return {"item_id": item_id}

@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    
    print(model_name,ModelName.lenet.value)
    
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

@app.get("/files/{file_path:path}") # esto no es muy recomendable usar
async def read_file(file_path: str):
    return {"file_path": file_path}

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# query parameters by default
# @app.get("/items/")
# async def read_item(skip: int = 0, limit: int = 10):
#     return fake_items_db[skip : skip + limit]

from typing import Union
#  q: Union[str, None] = None its a optional parameter
# @app.get("/items/{item_id}")
# async def read_item(item_id: str, q: Union[str, None] = None):
#     if q:
#         return {"item_id": item_id, "q": q}
#     return {"item_id": item_id}
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

# request body
@app.post("/itemsRequest/")
async def create_item(item: Item):
    return item

@app.get("/itemsValidationQuery/")
async def read_items(q: Union[str, None] = Query(default="fixedquery", max_length=50, min_length=3, regex="^fixedquery$")):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

@app.post("/files/")
async def create_file(file: bytes = File(description="A file read as UploadFile")): # esto es muy cutre
    return {"file_size": len(file)}

from io import BytesIO
from PIL import Image
from numpy import asarray

# ojo jpg no soporta RGBA pero png si
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile ):
    content = await file.read()
    im = Image.open(BytesIO(content)).convert("RGB")
    # im_arr = asarray(im)
    # print(im_arr.shape)
    
    
    return {"filename": file.filename, "contentType": file.content_type}


from fastapi.responses import FileResponse
import os
from starlette.background import BackgroundTasks
def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/sendImage")
async def sendImage(file: UploadFile,background_tasks: BackgroundTasks):
    content = await file.read()
    im = Image.open(BytesIO(content)).convert("RGB")
    im.save("tmp/test.jpg")
    background_tasks.add_task(remove_file, "tmp/test.jpg")
    return FileResponse("tmp/test.jpg")




