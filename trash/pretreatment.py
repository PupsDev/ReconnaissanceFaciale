# import the opencv library
import cv2


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#n nombre image
# n = 100
# for i in range(1,n+1):
#     #print(format(i, '03d'))
#     frame = cv2.imread('ressource/dataset/dataset/0'+format(i, '03d')+'.jpg') 
#     #print(frame)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     #for (x, y, w, h) in faces:q
#     #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     #(x, y, w, h) = faces    
#     #
#     #print(faces[0])
#     hmax =-1
#     indice = -1
#     ind = 0
#     offset =0
#     for face in faces:
#         if (face[2]+face[3]) > hmax :
#             hmax = face[2]+face[3]
#             indice = ind
#         ind+=1
#     #print(indice)
#     if i == 11 :
#         indice = 0


#     x= faces[indice][0]
#     y =faces[indice][1]
#     w=faces [indice][2]
#     h=faces [indice][3]

#     if i == 85 or i == 88 or i == 79:
#        offset = h//2 
#     #cv2.imshow('frame', frame)
#     frame = frame[y+offset:y+h+offset//4, x:x+w]
#     frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
#     cv2.imwrite("ressource/dataset/croppedfaces/face"+format(i, '03d')+".jpg", frame)
#     # Display the resulting frame
#     #cv2.imshow('frame', frame)
    

#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

i = 88
frame = cv2.imread('ressource/dataset/dataset/0'+format(i, '03d')+'.jpg') 
#print(frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#for (x, y, w, h) in faces:q
#    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#(x, y, w, h) = faces    
#
#print(faces[0])
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

if i == 85 or i == 88 or i == 79:
    offset = h//2 
#cv2.imshow('frame', frame)
frame = frame[y+offset:y+h+offset//2, x+offset//4:x+w-offset//4]
frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
cv2.imwrite("ressource/dataset/croppedfaces/face"+format(i, '03d')+".jpg", frame)
# After the loop release the cap object
#vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
