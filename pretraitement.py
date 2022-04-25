# import the opencv library
import cv2
import math
from PIL import Image
import numpy as np


# define a video capture object
#vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
nose_detector = cv2.CascadeClassifier("nose.xml")
i = 88
offs = 0


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def getEyes(frame):
    eyes = eye_detector.detectMultiScale(frame, 1.1, 4)

    index = 0
    eye_1 = [None, None, None, None]
    eye_2 = [None, None, None, None]

    diff = 200
    # print(eyes)

    eyes = [eye for eye in eyes if eye[2]+eye[3] > 70]

    while len(eyes) > 1 and abs(eyes[0][1]-eyes[1][1]) > 50:
        y = min(eyes[0][1], eyes[1][1])
        eyes = [eye for eye in eyes if eye[1] > y]
    # print(eyes)
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        #cv2.rectangle(frame,(eye_x, eye_y),(eye_w+eye_x, eye_y+eye_h), (255,0,0),2)
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)

        index = index + 1

    detected = False
    if (eye_1[0] is not None) and (eye_2[0] is not None):
        # print(eye_1)
        # print(eye_2)
        detected = True
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        #cv2.rectangle(frame,(eye_1[0], eye_1[1]),(eye_1[0]+eye_1[2], eye_1[1]+eye_1[3]), (255,0,0),2)
        #cv2.rectangle(frame,(eye_2[0], eye_2[1]),(eye_2[0]+eye_2[2], eye_2[1]+eye_2[3]), (255,0,0),2)

        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        #cv2.circle(frame, left_eye_center, 2, (255, 0, 0) , 2)
        #cv2.circle(frame, right_eye_center, 2, (255, 0, 0) , 2)
        #cv2.line(frame,right_eye_center, left_eye_center,(67,67,67),2)

        if left_eye_y < right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            #print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            #print("rotate to inverse clock direction")

        #cv2.circle(frame, point_3rd, 2, (255, 0, 0) , 2)

        #cv2.line(frame,right_eye_center, left_eye_center,(67,67,67),2)
        #cv2.line(frame,left_eye_center, point_3rd,(67,67,67),2)
        #cv2.line(frame,right_eye_center, point_3rd,(67,67,67),2)

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)

        xtranslate = frame.shape[1]/2 - \
            (right_eye_center[0]+left_eye_center[0])/2
        ytranslate = frame.shape[0]/2 - \
            (right_eye_center[1]+left_eye_center[1])/2

        middlex = (right_eye_center[0]+left_eye_center[0])/2
        middley = (right_eye_center[1]+left_eye_center[1])/2

        #cv2.circle(frame, (int(middlex),int(middley)), 10, (0, 0, 255) , 2)

        # print(xtranslate)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        #print("cos(a) = ", cos_a)

        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi
        if abs(right_eye_center[0] - left_eye_center[0]) < 100:
            detected = False
    if detected:
        return angle, (middlex, middley), direction
    else:
        return 0, (0, 0), 0


def align(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_detector.detectMultiScale(gray, 1.1, 4)

    index = 0
    eye_1 = [None, None, None, None]
    eye_2 = [None, None, None, None]

    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)

        index = index + 1
    detected = False
    if (eye_1[0] is not None) and (eye_2[0] is not None):
        detected = True
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        cv2.rectangle(frame, (eye_1[0], eye_1[1]), (eye_1[0] +
                      eye_1[2], eye_1[1]+eye_1[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (eye_2[0], eye_2[1]), (eye_2[0] +
                      eye_2[2], eye_2[1]+eye_2[3]), (255, 0, 0), 2)

        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        cv2.circle(frame, left_eye_center, 2, (255, 0, 0), 2)
        cv2.circle(frame, right_eye_center, 2, (255, 0, 0), 2)
        cv2.line(frame, right_eye_center, left_eye_center, (67, 67, 67), 2)

        if left_eye_y < right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            #print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            #print("rotate to inverse clock direction")

        cv2.circle(frame, point_3rd, 2, (255, 0, 0), 2)

        cv2.line(frame, right_eye_center, left_eye_center, (67, 67, 67), 2)
        cv2.line(frame, left_eye_center, point_3rd, (67, 67, 67), 2)
        cv2.line(frame, right_eye_center, point_3rd, (67, 67, 67), 2)

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)

        xtranslate = frame.shape[1]/2 - \
            (right_eye_center[0]+left_eye_center[0])/2
        ytranslate = frame.shape[0]/2 - \
            (right_eye_center[1]+left_eye_center[1])/2

        middlex = (right_eye_center[0]+left_eye_center[0])/2
        middley = (right_eye_center[1]+left_eye_center[1])/2

        cv2.circle(frame, (int(middlex), int(middley)), 10, (0, 0, 255), 2)

        # print(xtranslate)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        #print("cos(a) = ", cos_a)

        angle = np.arccos(cos_a)
        #print("angle: ", angle," in radian")

        angle = (angle * 180) / math.pi
        print("angle: ", angle, " in degree")
        if direction == -1:
            angle = 90 - angle
        direction *= -1
        new_img = Image.fromarray(frame)
        new_img = new_img.transform(
            new_img.size, Image.AFFINE, (1, 0, -xtranslate, 0, 1, -ytranslate))
        if angle > 15 and angle < 45:
            new_img = np.array(new_img.rotate(direction * angle))
        else:
            new_img = np.array(new_img)

    if detected:
        return new_img
    else:
        return frame


def findNose(gray):
    noses = nose_detector.detectMultiScale(gray)
    for (x, y, w, h) in noses:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imwrite("ressource/dataset/database2/faceNose.jpg", gray)


def pretreatment(frame):
    print("searching nose")
    findNose(frame)
    print("Aligning")
    aligned_image = align(frame)
    return aligned_image


def crop(frame):
    #print("detecting face")
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    hmax = -1
    indice = -1
    ind = 0
    offset = 0
    for face in faces:
        if (face[2]+face[3]) > hmax:
            hmax = face[2]+face[3]
            indice = ind
        ind += 1
    # for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if indice != -1:
        x = faces[indice][0]
        y = faces[indice][1]
        w = faces[indice][2]
        h = faces[indice][3]

        if w > 1000:
            w = 700
            x += 100
        if h > 1000:
            h = 700
            y += 100
        angle, eyesCenter, direction = getEyes(frame)

        new_img = Image.fromarray(frame)
        #new_img = new_img.transform(new_img.size, Image.AFFINE, (1, 0, 0, 0, 1, 100))
        if eyesCenter != (0, 0):
            translatex = w/2 - eyesCenter[0]+x
            translatey = h/2 - eyesCenter[1]+y
            # print(w)
            # print(h)

            #new_img = new_img.transform(new_img.size, Image.AFFINE, (1, 0, -translatex, 0, 1, -translatey))

            print(angle)
            if angle < 45:
                print("rotate")
                new_img = np.array(new_img.rotate(-1*direction * angle))

        frame = np.array(new_img)
        frame = frame[y+offset:y+h+offset//2, x+offset//4:x+w-offset//4]

    else:
        print(indice)
        frame = frame
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    return blur


def cropUnrotated(frame):
    print("detecting face")
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    hmax = -1
    indice = -1
    ind = 0
    offset = 0
    for face in faces:
        if (face[2]+face[3]) > hmax:
            hmax = face[2]+face[3]
            indice = ind
        ind += 1
    # for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if indice != -1:
        x = faces[indice][0]
        y = faces[indice][1]
        w = faces[indice][2]
        h = faces[indice][3]

        if w > 1000:
            w = 700
            x += 100
        if h > 1000:
            h = 700
            y += 100
        angle, eyesCenter, direction = getEyes(frame)
        if eyesCenter != (0, 0):
            translatex = w/2 - eyesCenter[0]+x
            translatey = h/2 - eyesCenter[1]+y
            # print(w)
            # print(h)
            new_img = Image.fromarray(frame)
            print("translate")
            #new_img = new_img.transform(new_img.size, Image.AFFINE, (1, 0, -translatex, 0, 1, -(translatey)))

            frame = np.array(new_img)
        frame = frame[y+offset:y+h+offset//2, x+offset//4:x+w-offset//4]

    else:
        print(indice)
        frame = frame
    return frame


def save(frame, i, j, name):
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite("ressource/dataset/treated/database2/face" +
                format(i, '02d')+'_'+format(j, '02d')+name+".jpg", frame)


def save1(frame, i, name):
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite("ressource/dataset/treated/database2/face"+format(i //
                6+1, '02d')+'_'+format(i % 6, '02d')+name+".jpg", frame)


def open(set, i, j):

    frame = cv2.imread('ressource/dataset/untreated/dataset' +
                       str(set)+'/face'+format(i, '02d')+"_"+format(j, '02d')+'.jpg')

    framer = crop(frame)
    save(framer, i, j, "")

    #frame2 = cropUnrotated(frame)
    # save(frame2,i,j,"2")


def open1(set, i):

    frame = cv2.imread(
        'ressource/dataset/untreated/dataset/'+format(i, '04d')+'.jpg')

    #test =crop(frame)
    #test = pretreatment(test)
    #save1(test,i, "2")
    framer = crop(frame)
    save1(framer, i, "")

    #frame2 = cropUnrotated(frame)
    # save1(frame2,i,"2")


def loadDataset3():
    for i in range(1, 20):
        if i in [3, 10, 16]:
            for j in range(6, 26):
                open(3, i, j)
        elif i != 8:
            for j in range(16, 26):
                open(3, i, j)


def loadDataset1():
    for i in range(90, 90):
        open1(1, i)


loadDataset1()
# loadDataset3()

cv2.destroyAllWindows()
