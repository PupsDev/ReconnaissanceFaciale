import cv2
import glob

path = "ressource/dataset/database25Blurred/*.jpg"

for index, file in enumerate(glob.glob(path)):
    # print(index)
    # print(file)
    a = cv2.imread(file)
    blur = cv2.GaussianBlur(a, (3, 3), 0)
    # print("ressource/dataset/database25Blurred2/"+file+".jpg")
    cv2.imwrite(file, blur)
