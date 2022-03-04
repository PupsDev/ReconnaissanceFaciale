import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import cv2 

n = 37

imgs =[]
#revoir calcul dimension pour tenir dans la mémoire en local

for i in range(n):
    img =cv2.imread('yale/face'+str(i)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_percent = 25 # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    gray = np.ndarray.flatten(resized)
    imgs.append(gray)
    #np.stack(imgs, gray, axis=1)


numpy_array = np.array(imgs)
transpose = numpy_array.T

# On transpose pour avoir un tableau de  colonne : n images  et lignes : p*q pixels
#
#

U, S, VT = np.linalg.svd(transpose,full_matrices=0)
print(S)
fig1 = plt.figure()
ax2 = fig1.add_subplot(122)
img_u1 = ax2.imshow(np.reshape(U[:,0],(192//4,168//4)))
img_u1.set_cmap('gray')
plt.axis('off')
plt.show()

r = 25
testFaceMS = transpose[:,0]
reconFace =  U[:,:r]  @ U[:,:r].T @ testFaceMS
img = plt.imshow(np.reshape(reconFace,(192//4,168//4)))
img.set_cmap('gray')
plt.title('r = ' + str(r))
plt.axis('off')
plt.show()