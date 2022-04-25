import cv2
import glob

path = "ressource/dataset/aligned/dataset/0001.jpg"

image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

cv2.imshow("gray", gray)
cv2.imshow("egalise", equalized)
cv2.waitKey(0)
