# -*- coding: utf-8 -*-
"""ProjetFini.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14yIHtfVHZIpzh_tf7mSPNx6GH_FIsY0l
"""

# # Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import cv2
import sys

n = 99
noms = ["gauthier","albena","mathieu","alexandre F", "dorian","thomas ?", "erwan","ange","roland","aurelien","samuel","alexandre S","florentin","sylvain","khélian","camille","marius","alexandre L","thomas S","maxime"]
def loadImages():
    """On charge notre dataset des étudiants d'imagine en gardant les (n-1) premiers donc 95 images sur 100 images. (Il y a 5 images par étudiant)"""

    

    imgs =[]

    #face 1 à n+1
    for i in range(1,n+1):
        stri = 'ressource/dataset/croppedfaces64/face'+format(i, '03d')+'.jpg'
        img =cv2.imread(stri)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.ndarray.flatten(gray)
        imgs.append(gray)

    # n = 2400
    # #face 1 à n+1
    # for i in range(1,n+1):
    #     stri = 'yale/all/all/face'+str(i)+'.jpg'
    #     img =cv2.imread(stri)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = np.ndarray.flatten(gray)
    #     imgs.append(gray)
    return imgs

def preProcess(imgs):
    """On transforme notre tableau d'image en numpy array et on enlève la moyenne des visage afin de centrer notre nuage de point et de pouvoir calculer une ACP. """

    numpy_array = np.array(imgs)
    average = np.mean(numpy_array,axis=0)
    numpy_array = numpy_array - np.tile(average,(numpy_array.shape[0],1))
    
    return numpy_array, average
def load(filepath):
    img =cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testFaceMS = np.ndarray.flatten(gray)-average
    return testFaceMS

def display(reconFace):
    img = plt.imshow(np.reshape(reconFace,(64,64)))
    img.set_cmap('gray')
    plt.title('r = ' + str(r))
    plt.axis('off')
    plt.show()


def computeDistances(testFaceMS,r):
    """Dernière étape on calcule les poids de chaque image du dataset et on calcule la distance du visage inconnu à celui de tous les visages connus.

    """
    seuil = 900
    w = testFaceMS @ U[:,:r]
    weights = []
    distances =[]
    for ind,image in enumerate(numpy_array) :
        weight = image @ U[:,:r]
        dist = np.linalg.norm(weight-w)
        distances.append((dist,ind))
        #if dist<seuil :
            #print(str(ind+1)+":"+str(dist))
        #if ind == 10 :
        #print("--> "+str(ind+1)+":"+str(dist))
        weights.append(weight)

    #dist = np.linalg.norm(weights[10]-weights[90])
    
    return distances

def findSeuil(testFaceMS,r):
    seuil = 900
    w = testFaceMS @ U[:,:r]
    weights = []
    distances =[]
    for ind,image in enumerate(numpy_array) :
        weight = image @ U[:,:r]
        dist = np.linalg.norm(weight-w)
        distances.append((dist,ind))
        #if dist<seuil :
            #print(str(ind+1)+":"+str(dist))
        #if ind == 10 :
        #print("--> "+str(ind+1)+":"+str(dist))
        weights.append(weight)
    
    dist2=[]
    res =0
    i = 1
    k = 0
    for dist,ind in distances:
        if i < 6 :
            res +=dist
            i+=1
        if i ==6 :
            res/=5
            dist2.append((res,k))
            k+=1
            res =0
            i = 1
            
    return dist2
def printByName(liste):
    for dist,ind in liste:
        #rint(ind)
        print(noms[ind]+"->"+str(dist))
def printByName2(liste):
    for dist,ind in liste:
        #print(ind)
        print(noms[ind//5 ]+"->"+str(dist))

def test(id):
    filepath = "ressource/dataset/croppedfaces64/face"+format(id, '03d')+".jpg"
    img =cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testFaceMS = np.ndarray.flatten(gray)-average
    return testFaceMS

def computeImage(id, r, k,seuil):
    
    testFaceMS = test(id)
    #display(testFaceMS)
    reconFace =  average  + U[:,:r]  @ U[:,:r].T @ testFaceMS
    #display(reconFace)

    distances = computeDistances(testFaceMS,r)
    distances.sort()

    #dist2 = findSeuil(testFaceMS,r)
    #print(distances[:10])
    #dist2.sort()
    #print(dist2)
    #print("Solo")

    #print("\n Group \n")
    #printByName(dist2[:5])
    #print("\n")
    #print("\n")

    distance = [(a,b) for a,b in distances if a < seuil]
    #printByName2(distance)
    VP = 0
    FP = 0

    for dist,ind in distance:
        if dist < seuil:
            if k == ind//5:
                VP+=1
            else :
                FP+=1
    
    return VP,FP



imgs = loadImages()
numpy_array,average = preProcess(imgs)
transpose = numpy_array.T
U, S, VT = np.linalg.svd(transpose,full_matrices=0)

# filepath = 'ressource/dataset/indianFlorentin.jpg'
# testFaceMS = load(filepath)

r = 6

P =5
sens = open("sens.dat", "w")
spec = open("spec.dat", "w")

test2 = open("test.dat", "w")
ind = 0

for seuil in range(500,5500,250):
    scoreVP = []
    scoreFP = []
    print("seuil="+str(seuil))
    for k in range(20):
        sumFP = 0
        sumVP = 0
        #print("test de "+str(k)+"\n")
        for i in range( (5*k)+1,(5*k)+6):
            VP,FP =computeImage(i,r,k,seuil)
            sumVP +=VP
            sumFP +=FP
        #print( (sumVP,sumFP))
        #scoreFP.append((noms[k],sumFP))
        scoreVP.append(sumVP)
        scoreFP.append(sumFP)

    VP =sum(i for i in scoreVP )
    FP =sum(i for i in scoreFP )

    FPR = FP/(10000-500)
    sensi =VP/(100*P)
    speci = 1-FPR
    print("sensivity : " + str(sensi))
    print("specificity: "+str(speci))
    sens.write(str(seuil)+" "+str(sensi)+"\n")
    spec.write(str(seuil)+" "+str(speci)+"\n")
    
    test2.write(str(seuil)+" "+str(speci+sensi)+"\n")

#print(#VP)
#print(FP)

#print("sensivity : " scoreVP/P)
# r = 6
# reconFace =  average  + U[:,:r]  @ U[:,:r].T @ testFaceMS
# display(reconFace)

# distances = computeDistances(testFaceMS,r)
# distances.sort()
# printByName2(distances[:5])

