import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import sys
import cv2

n = 94

imgs =[]
#revoir calcul dimension pour tenir dans la mémoire en local
i =5

#face 1 a 99
for i in range(1,n+1):
    stri = 'ressource/dataset/croppedfaces64/face'+format(i, '03d')+'.jpg'
    img =cv2.imread(stri)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.ndarray.flatten(gray)
    imgs.append(gray)


numpy_array = np.array(imgs)
average = np.mean(numpy_array,axis=0)
numpy_array = numpy_array - np.tile(average,(numpy_array.shape[0],1))
transpose = numpy_array.T

U, S, VT = np.linalg.svd(transpose,full_matrices=0)

fig1 = plt.figure()
ax2 = fig1.add_subplot(122)
img_u1 = ax2.imshow(np.reshape(U[:,0],(64,64)))
img_u1.set_cmap('gray')
plt.axis('off')
plt.show()

r = n
testFaceMS = transpose[:,n-1]

# 100 n'est pas dans le set
# stri = 'ressource/dataset/croppedfaces64/face100.jpg'
# img =cv2.imread(stri)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# testFaceMS = np.ndarray.flatten(gray)-average


reconFace =  average + U[:,:r]  @ U[:,:r].T @ testFaceMS
img = plt.imshow(np.reshape(reconFace,(64,64)))
img.set_cmap('gray')
plt.title('r = ' + str(r))
plt.axis('off')
plt.show()

w = testFaceMS @ U[:,:r]
weights = []
for image in numpy_array :
    weight = image @ U[:,:r]
    dist = np.linalg.norm(weight-w)
    print(dist)
    weights.append(weight)
# print(weights[-1])
# print(w)

