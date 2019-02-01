from keras.datasets import cifar10
from PIL import Image
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

max_num_datas = 50 # 트레이닝 이미지 개수
test_num_datas=10 # 테스트 이미지 개수
num_classes = 4
num_datas_list = np.zeros(num_classes)

img_dir = "data" # 트레이닝 이미지 저장 폴더
img_dirt = "datat" # 테스트 이미지 저장 폴더

id = 0

for x, y in zip(x_train, y_train):

    if np.sum(num_datas_list) > (max_num_datas +test_num_datas)* len(num_datas_list):
        break

    label = y[0]
    if label >= num_classes:
        continue

# download training data
    if num_datas_list[label] <max_num_datas:
        num_datas_list[label] += 1

        img_path = os.path.join(img_dir, "{}_{}.jpg".format(label, id))
        id += 1
        img = Image.fromarray(x)
        img.save(img_path)

    # download testing data
    if num_datas_list[label]>=max_num_datas and num_datas_list[label] < (max_num_datas+test_num_datas ):
        num_datas_list[label] += 1

        img_path = os.path.join(img_dirt, "{}_{}.jpg".format(label, id))
        id += 1
        img = Image.fromarray(x)
        img.save(img_path)

    if num_datas_list[label] >= max_num_datas + test_num_datas:
        continue



