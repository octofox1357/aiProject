from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
import joblib
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def load_images_from_folder(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = np.array(cv2.resize(img, dsize=(100, 100),
                           interpolation=cv2.INTER_AREA))
            images = np.append(images, img)
    images = images.reshape(-1, 100*100)
    return images


images = load_images_from_folder('C:/Users/john/Desktop/python/images')
print(images.shape)
test_img = images[:5]
print(test_img)
targetArr = np.array(['1', '2', '3', '4', '5', '6',
                     '7', '8', '9', '10', '11', '12', '13', '14'])

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(images, targetArr)
print(kn.predict(test_img))

# 모델을 파일로 저장하기
# joblib.dump(kn, '../ai/skin/imageClassification.pkl')
# joblib.dump(kn, '../ai/scalp/imageClassification.pkl')
# joblib.dump(kn, '../ai/mock/imageClassification.pkl')


# 모델을 파일로 불러오기
# clf_from_joblib = joblib.load('./imageClassification.pkl')
# print(clf_from_joblib.predict(test_img))
