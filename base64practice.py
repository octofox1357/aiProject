from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel
import base64
from PIL import Image
import cv2
import numpy as np
import io
import joblib

app = FastAPI(title="skin oil ML API",
              description="API for skin oil dataset ml model", version="1.0")

model = None  # 인공지능 변수 전역에 선언

@app.on_event('startup')  # 서버 시작과 함께 인공지능 모델을 불러옴
async def load_model():
    global model
    model = joblib.load('../ai/mock/imageClassification.pkl')


@app.post("/predict/skin/oil")
def skin(base64_string: str):
    if(base64_string == None):
        return {
            "result": False,
            "Error": "Data input required"
        }
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(image), cv2.IMREAD_GRAYSCALE)
    img = np.array(cv2.resize(img, dsize=(256, 256),
                              interpolation=cv2.INTER_AREA))
    img = img.reshape(-1, 256*256)
    global model
    return {"result": str(model.predict(img)[0])}
