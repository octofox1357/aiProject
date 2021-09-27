from fastapi import FastAPI, File, UploadFile, Path, Request, Form
from fastapi.responses import HTMLResponse
from typing import List, Dict, Optional  # 타입을 지원하는 파이썬의 스탠다드 라이브러리임
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image
import io
import joblib

app = FastAPI(title="skin oil ML API",
              description="API for skin oil dataset ml model", version="1.0")

model = None  # 인공지능 변수 전역에 선언


@app.on_event('startup')  # 서버 시작과 함께 인공지능 모델을 불러옴
async def load_model():
    global model
    model = joblib.load('../ai/skin/test/imageClassification.pkl')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/skin/oil")
async def main():
    content = """
        <body>
        <div>
        <h6>muliple</h6>
        <form action="/predict/skin/oil/files" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>

        <h6>same name</h6>
        <form action="/predict/skin/oil/files" enctype="multipart/form-data" method="post">
        <input name="file1" type="file" >
        <input name="file2" type="file" >
        <input name="file3" type="file" >
        <input type="submit">
        </form>
        </div>
        </body>
    """
    return HTMLResponse(content=content)

# https://stackoverflow.com/questions/61333907/receiving-an-image-with-fast-api-processing-it-with-cv2-then-returning-it

@app.post("/predict/skin/oil")
async def predict_skin_oil(files: List[UploadFile] = File(...)):
    if(files == None):
        return {
            "result": False,
            "Error": "Data input required"
        }
    resArr = []
    for file in files:
        contents = await file.read()
        npArr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
        img = np.array(cv2.resize(img, dsize=(256, 256),
                                  interpolation=cv2.INTER_AREA))
        img = img.reshape(-1, 256*256)
        global model
        resArr.append(model.predict(img)[0])
    return {"result": resArr}


@app.post("/predict/skin/oil/files")
async def predict_skin_oil(file1: Optional[UploadFile] = File(None), file2: Optional[UploadFile] = File(None), file3: Optional[UploadFile] = File(None), file4: Optional[UploadFile] = File(None),
                           file5: Optional[UploadFile] = File(None), file6: Optional[UploadFile] = File(None), file7: Optional[UploadFile] = File(None),
                           file8: Optional[UploadFile] = File(None), file9: Optional[UploadFile] = File(None), file10: Optional[UploadFile] = File(None)):
    if(file1.filename == ""):
        return {
            "result": False,
            "Error": "Data input required"
        }

    fileArr = []
    if file1.filename != "":
        fileArr.append(file1)
    if file2.filename != "":
        fileArr.append(file2)
    if file3.filename != "":
        fileArr.append(file3)
    if file4.filename != "":
        fileArr.append(file4)
    if file5.filename != "":
        fileArr.append(file5)
    if file6.filename != "":
        fileArr.append(file6)
    if file7.filename != "":
        fileArr.append(file7)
    if file8.filename != "":
        fileArr.append(file8)
    if file9.filename != "":
        fileArr.append(file9)
    if file10.filename != "":
        fileArr.append(file10)

    resArr = []
    for file in fileArr:
        contents = await file.read()
        npArr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)
        img = np.array(cv2.resize(img, dsize=(256, 256),
                                  interpolation=cv2.INTER_AREA))
        img = img.reshape(-1, 256*256)
        global model
        resArr.append(model.predict(img)[0])
    return {"result": resArr}
