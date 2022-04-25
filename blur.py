import cv2
import glob

path = "ressource/dataset/aligned/dataset3_blurred/*.jpg"

for index, file in enumerate(glob.glob(path)):
    # print(index)
    # print(file)
    image = cv2.imread(file)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # equalized = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(image, (3, 3), 0)

    cv2.imwrite(file, blur)
