# import the opencv library
import cv2


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


i = 88
offs = 0

def open(i ,j):

        frame = cv2.imread('ressource/dataset/dataset3/face'+format(i, '02d')+"_"+format(j, '02d')+'.jpg')
            
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

        if indice != -1 :
            x= faces[indice][0]
            y =faces[indice][1]
            w=faces [indice][2]
            h=faces [indice][3]
            frame = frame[y+offset:y+h+offset//2, x+offset//4:x+w-offset//4]

        else:
            print(indice)

        frame = cv2.resize(frame, (64,64), interpolation = cv2.INTER_AREA)

        #print(i)

        #print(format(i, '02d')+'_'+format(j, '02d'))

        cv2.imwrite("ressource/dataset/database25/face"+format(i, '02d')+'_'+format(j, '02d')+".jpg", frame)
for i in range(1,20):
    if i in [3,10,16]:
        for j in range(6,26) :
            open(i ,j)
    elif i != 8:
        for j in range(16,26) :
            open(i ,j)


    
cv2.destroyAllWindows()
