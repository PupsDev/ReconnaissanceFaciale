# import the opencv library
import cv2
from skimage import exposure
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


for i in range(1, 65):
    frame = cv2.imread('yale/all/face'+str(i)+'.jpg')
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    for j in range(1, 20):
        test = cv2.imread('ressource/dataset/new/aligned_equalized_blurred_spec/face' +
                          format(j, '02d')+'_01.jpg')
        #blur2 = cv2.GaussianBlur(test,(5,5),0)
        # print(frame)
        matched = exposure.match_histograms(test, frame, channel_axis=-1)

        start = 25

        num = i + start
        # if j == 8:
        print("test "+str(j))
        cv2.imwrite('ressource/dataset/new/aligned_equalized_blurred_spec/face' +
                    format(j, '02d')+'_' + format(num, '02d')+'.jpg', matched)

cv2.destroyAllWindows()
