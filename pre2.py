# import the opencv library
import cv2


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


for i in range(1,96):
    frame = cv2.imread('ressource/dataset/dataset/'+format(i, '04d')+'.jpg') 
    #print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    hmax =-1
    indice = -1
    ind = 0
    offset =0
    for face in faces:
        if (face[2]+face[3]) > hmax :
            hmax = face[2]+face[3]
            indice = ind
        ind+=1
    #print(indice)

    if i == 11 :
        indice = 0


    x= faces[indice][0]
    y =faces[indice][1]
    w=faces [indice][2]
    h=faces [indice][3]

    if i == 80 or i == 83 or i == 74:
        offset = h//2 
    #cv2.imshow('frame', frame)
    frame = frame[y+offset:y+h+offset//2, x+offset//4:x+w-offset//4]
    frame = cv2.resize(frame, (64,64), interpolation = cv2.INTER_AREA)

    cv2.imwrite("ressource/dataset/database5/face"+format((i-1)//5 +1, '02d')+'_'+format((i-1)%5 +1, '02d')+".jpg", frame)

cv2.destroyAllWindows()
