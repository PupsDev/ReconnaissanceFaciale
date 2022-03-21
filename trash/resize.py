import cv2


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#n nombre image
n = 100
for i in range(1,n+1):
    #print(format(i, '03d'))
    frame = cv2.imread('ressource/dataset/croppedfaces256/face'+format(i, '03d')+'.jpg') 

    frame = cv2.resize(frame, (64,64), interpolation = cv2.INTER_AREA)
    cv2.imwrite("ressource/dataset/croppedfaces64/face"+format(i, '03d')+".jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break