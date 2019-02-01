# At first, install the opencv through this command in cmd
# pip install OpenCV-Python

import os
import cv2

path_1 = "image/zz" #collected image class 1
path_2 = "image/zb" #collected image class 2
path_3 = "image/ts" #collected image class 3
path_t="data" #save renamed images

id=0
IMG_SIZE = 32
# rename as 0_*.jpg, 1_*.jpg, 2_*.jpg
# for (path, dirs, files) in os.walk(path_1):
#     for filename in files:
#         newname = "0_{}.jpg".format(id)
#         id+=1
#         os.rename(path_1 + "\\" + filename, path_t + "\\" + newname)
#
# for (path, dirs, files) in os.walk(path_2):
#     for filename in files:
#         newname = "1_{}.jpg".format(id)
#         id+=1
#         os.rename(path_2 + "\\" + filename, path_t + "\\" + newname)
#
# for (path, dirs, files) in os.walk(path_3):
#     for filename in files:
#         newname = "2_{}.jpg".format(id)
#         id+=1
#         os.rename(path_3 + "\\" + filename, path_t + "\\" + newname)

# resize the training images
# for filename in os.listdir(path_t):
#     print(filename)
#     path= "data"+ "\\" +filename
#     img = cv2.imread(path)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     cv2.imwrite("resize_data"+ "\\"+filename,img)

# resize the test images
# for filename in os.listdir("test"):
#     print(filename)
#     path= "test"+ "\\" +filename
#     img = cv2.imread(path)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     cv2.imwrite("resize_test"+ "\\"+filename,img)
