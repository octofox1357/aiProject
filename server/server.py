from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import HTMLResponse
from typing import List, Optional # 타입을 지원하는 파이썬의 스탠다드 라이브러리임
from pydantic import BaseModel
import numpy as np
import cv2
import joblib

app = FastAPI(title="skin oil ML API",
              description="API for skin oil dataset ml model", version="1.0")

model = None  # 인공지능 변수 전역에 선언

@app.on_event('startup')  # 서버 시작과 함께 인공지능 모델을 불러옴
async def load_model():
    global model
    model = joblib.load('./imageClassification.pkl')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/skin/oil")
async def main():
    content = """
        <body>
        <form action="/predict/skin/oil" enctype="multipart/form-data" method="post">
        <input name="file" type="file" multiple>
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)

# https://stackoverflow.com/questions/61333907/receiving-an-image-with-fast-api-processing-it-with-cv2-then-returning-it
@app.post("/predict/skin/oil")
async def predict_skin_oil(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = np.array(cv2.resize(img, dsize=(100, 100),
                              interpolation=cv2.INTER_AREA))
    img = img.reshape(-1, 100*100)
    global model
    model.predict(img)
    return {"result": str(model.predict(img)[0])}
