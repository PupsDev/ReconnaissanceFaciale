# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import argparse
import cv2
from PIL import Image, ImageTk
import os

# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image tests/thomas.jpg

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
rects = detector(gray, 2)
# test_rect = rects[0]
# print(len(rects))
# loop over the face detections

rect = rects[0]
(x, y, w, h) = rect_to_bb(rect)
faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
faceAligned = fa.align(image, gray, rect)
# faceAligned = cv2.resize(faceAligned, (64, 64),
#                          interpolation=cv2.INTER_AREA)
# display the output images
cv2.imshow("Original", faceOrig)
cv2.imshow("Aligned", faceAligned)
#     print("ressource/dataset/database25Aligned/" +
#           os.path.basename(args["image"]))
# cv2.imwrite("ressource/dataset/untreated/dataset_crop/" +
#             os.path.basename(args["image"]), faceAligned)
cv2.waitKey(0)
