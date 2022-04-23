# import the opencv library
import cv2
import math
from PIL import Image
import numpy as np

# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")


i = 88
offs = 0
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

frame = cv2.imread('ressource/dataset/dataset/0090.jpg') 

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


eyes = eye_detector.detectMultiScale(gray)

index = 0
eye_1 = [None, None, None, None]
eye_2 = [None, None, None, None]
for (eye_x, eye_y, eye_w, eye_h) in eyes:
    if index == 0:
        eye_1 = (eye_x, eye_y, eye_w, eye_h)
    elif index == 1:
        eye_2 = (eye_x, eye_y, eye_w, eye_h)

    cv2.rectangle(frame,(eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), (255,0,0),2)
    index = index + 1

if (eye_1[0] is not None) and (eye_2[0] is not None):
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    
    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    
    cv2.circle(frame, left_eye_center, 2, (255, 0, 0) , 2)
    cv2.circle(frame, right_eye_center, 2, (255, 0, 0) , 2)
    cv2.line(frame,right_eye_center, left_eye_center,(67,67,67),2)

    if left_eye_y < right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        print("rotate to inverse clock direction")
    
    cv2.circle(frame, point_3rd, 2, (255, 0, 0) , 2)
    
    cv2.line(frame,right_eye_center, left_eye_center,(67,67,67),2)
    cv2.line(frame,left_eye_center, point_3rd,(67,67,67),2)
    cv2.line(frame,right_eye_center, point_3rd,(67,67,67),2)

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    print("cos(a) = ", cos_a)
    
    angle = np.arccos(cos_a)
    print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    print("angle: ", angle," in degree")
    if direction == -1:
        angle = 90 - angle
    direction *=-1
    new_img = Image.fromarray(frame)
    new_img = np.array(new_img.rotate(direction * angle))
    cv2.imwrite("ressource/dataset/test.jpg", new_img)

#cv2.destroyAllWindows()
