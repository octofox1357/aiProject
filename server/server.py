from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import HTMLResponse
from typing import List, Optional # 타입을 지원하는 파이썬의 스탠다드 라이브러리임
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


class Base64Image(BaseModel):
    strData : str

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
        <form action="/predict/skin/oil" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)

# https://stackoverflow.com/questions/61333907/receiving-an-image-with-fast-api-processing-it-with-cv2-then-returning-it
@app.post("/predict/skin/oil")
async def predict_skin_oil(files: List[UploadFile] = File(...), alltotal = ''):
    if(files == None):
        return {
            "result": False,
            "Error": "Data input required"
        }
    resArr = [];
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


@app.post("/predict/skin/oil/base64")
def skin(imgStr: Base64Image):
    if(imgStr.strData == None):
        return {
            "result": False,
            "Error": "Data input required"
        }
    img_data = base64.b64decode(imgStr.strData)
    npimg = np.fromstring(img_data, dtype=np.uint8);
    source = cv2.imdecode(npimg, 1)
    img = cv2.cvtColor(source, cv2.IMREAD_GRAYSCALE)
    img = np.array(cv2.resize(img, dsize=(256, 256),
                              interpolation=cv2.INTER_AREA))
    img = img.reshape(-1, 256*256)
    global model
    return {"result": str(model.predict(img)[0])}
