from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import cv2
import os
from openpyxl import load_workbook

# data_only=True로 해줘야 수식이 아닌 값으로 받아온다.
load_wb = load_workbook(
    "C:/Users/john/Desktop/AIProject/data/skin/1/facehead.xlsx", data_only=True)
# 시트 이름으로 불러오기
load_ws = load_wb['Sheet1']

oilArr = np.array([]) # 유분값 리스트
filenameArr = np.array([]) # 이미지 이름 리스트

get_oil_cells = load_ws['B2': 'B26'] # 셀 범위 불러오기
get_filename_cells = load_ws['D2': 'D26']

# 유분값과 이미지를 각각의 배열에 저장
for row in get_oil_cells:
    for cell in row:
        oilArr = np.append(oilArr, cell.value)
        # print(cell.value)

for row in get_filename_cells:
    for cell in row:
        filenameArr = np.append(filenameArr, cell.value)
        # print(cell.value)


# 파일 이름의 배열로 이미지들을 배열에 로드함
def load_images_from_folder(folder, filenames):
    images = np.array([])
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename +'.jpg'), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = np.array(cv2.resize(img, dsize=(100, 100),
                           interpolation=cv2.INTER_AREA))
            images = np.append(images, img)
    images = images.reshape(-1, 100*100)
    return images


images = load_images_from_folder('C:/Users/john/Desktop/AIProject/data/skin/1', filenameArr)
# print(images.shape);
test_img = images[:5]

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(images, oilArr)
print(kn.predict(test_img))
joblib.dump(kn, 'C:/Users/john/Desktop/AIProject/ai/skin/test/imageClassification.pkl')

# 투두
# 엑셀에 명시된 폴더의 모든 이미지를 불러오고 길이 1000의 1차원 배열로 불러온다.
# [[ 1차원 이미지 ], [ 1차원 이미지 ], [ 1차원 이미지 ]...] (아웃풋은 2차원)
# 이미지의 수와 같은 타겟값(유분값) 배열을 준비한다.[15, 20, 18, 16 ...]
# 모델을 학습시키고 평가 점수를 확인한다.
# 적절한 점수가 나오면 해당 모델을 파일로 만들어서 저장하고 서버에서 사용한다.
# 최저치 15, 최고치 20, 현재는 5% 정도의 유분 차이만 존재한다.