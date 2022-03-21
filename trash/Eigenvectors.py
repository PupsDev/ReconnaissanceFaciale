import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2 
vmin = 0
vmax = 256
image_bias = 1 # sometimes 1
def plot_svd(A):
    n = 100
    #imshow(A, cmap='gray', vmin=vmin, vmax=vmax)
    #plt.show()
    U, S, V = svd(A)

    imgs = []
    sing = 0
    for i in range(n):
        img = S[i]*np.outer(U[:,i],V[i])
        imgs.append(img)
        sing = sing + S[i]
        print("Sing :" + str(S[i]))

    combined_imgs = []
    for i in range(0,n,3):
        img = sum(imgs[:i+1])
        combined_imgs.append(img)
        cv2.imwrite('result/combined'+str(i)+'.jpg', img)

    return U,S,V

img = cv2.imread('ressource/dataset/test.jpg') 
scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray = gray.flatten()
#print(gray)
U,S,V = plot_svd(gray)



