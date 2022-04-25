import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import os
import scipy.io
import cv2
import sys

from tkinter import *
# import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk

from functools import partial

# Image Alignment
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
# shape_predictor_68_face_landmarks.dat


class Reconnaissance:
    def __init__(self, path, croppedResolution, noms, imagePerPerson, personCount, n, equalizeAndBlur):
        self.pathDatabase = path
        self.fileNameAbsolute = ""
        self.fileNameRelative = ""
        self.imagesCount = n
        self.noms = noms
        self.croppedResolution = croppedResolution
        self.EqualizeAndBlur = equalizeAndBlur

        self.imagesDatabase = []
        self.imagesDatabaseArray = None
        self.averageImageDatabase = None
        self.imageCropped = None
        self.imageWebcam = None
        self.reconFace = None

        self.transpose = None
        self.U = None
        self.S = None
        self.VT = None
        self.r = int(0.15*n)
        self.seuil = 6000
        self.theta = 2000
        self.itheta = 2000
        self.removeJ = -1

        self.frameSpinbox = None
        self.slider = None
        self.currentValueDimension = None
        self.labelDimension = None
        self.spinBoxDimension = None
        self.currentValueSeuil = None
        self.labelSeuil = None
        self.spinBoxSeuil = None
        self.onlyOneSlider = False

        self.pauseVideoFlag = False
        self.imagePerPerson = imagePerPerson
        self.personCount = personCount

        self.acceptedListe = {}

        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")
        self.fa = FaceAligner(predictor, desiredFaceWidth=croppedResolution)
        self.starter()
        # self.getmaxdist()

    def switch_pauseFlag(self):
        self.pauseVideoFlag = not(self.pauseVideoFlag)
        # print(self.pauseVideoFlag)

    def starter(self):
        self.loadImages(self.personCount, self.imagePerPerson,
                        removeI=[], removeJ=self.removeJ)
        print("Preprocessing ...")
        self.preProcess()
        self.transpose = self.imagesDatabaseArray.T
        print("Calcul SVD ...")
        self.U, self.S, self.VT = np.linalg.svd(
            self.transpose, full_matrices=0)

        print("Done !")
        k = sum(self.S)
        liste = [100*i/k for i in self.S]
        var = []
        sumsum = 0
        for i in liste:
            sumsum += i
            var.append(sumsum)

    def getmaxdist(self):
        """Dernière étape on calcule les poids de chaque image du dataset et on calcule la distance du visage inconnu à celui de tous les visages connus."""

        weights = []
        distances = []
        for i, imageI in enumerate(self.imagesDatabaseArray):
            for j, imageJ in enumerate(self.imagesDatabaseArray):
                if i != j:
                    weight = imageI @ self.U[:, :self.r]
                    weight2 = imageJ @ self.U[:, :self.r]

                    dist = np.linalg.norm(weight-weight2)
                    weights.append(dist)
        print("MAX : ")
        self.theta = 0.34*max(weights)
        self.itheta = 0.66*max(weights)
        print(self.theta)

    def crop_filepath(self, filename):
        """On charge l'image du filepath et on la crop et on l'écrit en dur sur le disque """

        self.imageCropped = cv2.imread(filename)
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        hmax = -1
        indice = -1
        ind = 0
        offset = 0
        for face in faces:
            if (face[2]+face[3]) > hmax:
                hmax = face[2]+face[3]
                indice = ind
            ind += (1)
        if (indice >= 0):
            x = faces[indice][0]
            y = faces[indice][1]
            w = faces[indice][2]
            h = faces[indice][3]
        else:
            x = 0
            y = 0
            w = 64
            h = 64
        self.imageCropped = self.imageCropped[y:y+h, x:x+w]
        self.imageCropped = cv2.resize(
            self.imageCropped, (self.croppedResolution, self.croppedResolution), interpolation=cv2.INTER_AREA)
        # self.imageCropped = cv2.equalizeHist(self.imageCropped)

        # self.imageCropped = cv2.GaussianBlur(self.imageCropped, (3, 3), 0)
        # cv2.imwrite("cropped.jpg", self.imageCropped)

    def crop_webcam(self):
        """On charge l'image du filepath et on la crop et on l'écrit en dur sur le disque """

        self.imageCropped = self.imageWebcam
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        hmax = -1
        indice = -1
        ind = 0
        offset = 0
        for face in faces:
            if (face[2]+face[3]) > hmax:
                hmax = face[2]+face[3]
                indice = ind
            ind += (1)
        if (indice >= 0):
            x = faces[indice][0]
            y = faces[indice][1]
            w = faces[indice][2]
            h = faces[indice][3]
        else:
            x = 0
            y = 0
            w = self.croppedResolution
            h = self.croppedResolution
        self.imageCropped = self.imageCropped[y:y+h, x:x+w]
        self.imageCropped = cv2.resize(
            self.imageCropped, (self.croppedResolution, self.croppedResolution), interpolation=cv2.INTER_AREA)
        # cv2.imwrite("cropped.jpg", self.imageCropped)

    def crop_webcam_aligned(self):
        """On charge l'image du filepath et on la crop et on l'écrit en dur sur le disque """
        self.imageCropped = self.imageWebcam
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 2)
        if len(rects) > 0:
            # print("OK")
            rect = rects[0]
            (x, y, w, h) = rect_to_bb(rect)
            # print()
            self.imageCropped = self.fa.align(self.imageCropped, gray, rect)
            return True
        else:
            # self.crop_webcam_aligned()
            return False
            # self.imageCropped = self.imageCropped[0:self.croppedResolution,
            #                                       0:self.croppedResolution]
            # self.imageCropped = cv2.resize(
            #     self.imageCropped, (self.croppedResolution, self.croppedResolution), interpolation=cv2.INTER_AREA)

        # self.imageCropped = self.imageCropped[y:y+h, x:x+w]
        # self.imageCropped = cv2.resize(
        #     self.imageCropped, (self.croppedResolution, self.croppedResolution), interpolation=cv2.INTER_AREA)
        # cv2.imwrite("cropped.jpg", self.imageCropped)

    #######################################################################################################################################################################

    def displayImage(self, fileNameAbsolute, image_id, canvas):
        """Affiche l'image dans le canvas"""
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
        if self.EqualizeAndBlur:
            # print("OK")
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # create the image object, and save it so that it
        # won't get deleted by the garbage collector
        canvas.image_tk = ImageTk.PhotoImage(
            Image.fromarray(gray).resize((300, 300), Image.Resampling.LANCZOS))
        # configure the canvas item to use this image
        canvas.itemconfigure(image_id, image=canvas.image_tk)

    def loadImages(self, personCount, imagePerPerson, removeI, removeJ):
        """On charge notre dataset des étudiants d'imagine en gardant les (n-1) premiers donc 95 images sur 100 images. (Il y a 5 images par étudiant)"""
        print("Laoding database ..")

        for i in range(1, personCount+1):
            for j in range(1, imagePerPerson+1):
                stri = self.pathDatabase + \
                    format(i, '02d')+'_'+format(j, '02d')+'.jpg'
                # print(stri)
                img = cv2.imread(stri)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.ndarray.flatten(gray)
                self.imagesDatabase.append(gray)
        print("Database loaded !")

    def preProcess(self):
        """On transforme notre tableau d'image en numpy array et on enlève la moyenne des visage afin de centrer notre nuage de point et de pouvoir calculer une ACP. """
        self.imagesDatabaseArray = np.array(self.imagesDatabase)
        self.averageImageDatabase = np.mean(self.imagesDatabaseArray, axis=0)
        self.imagesDatabaseArray = self.imagesDatabaseArray - \
            np.tile(self.averageImageDatabase,
                    (self.imagesDatabaseArray.shape[0], 1))

    def load(self, imageCropped):
        gray = cv2.cvtColor(imageCropped, cv2.COLOR_BGR2GRAY)

        if self.EqualizeAndBlur:
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        self.testFaceMS = np.ndarray.flatten(gray)-self.averageImageDatabase

    #######################################################################################################################################################################
    def computeDistances(self):
        """Dernière étape on calcule les poids de chaque image du dataset et on calcule la distance du visage inconnu à celui de tous les visages connus."""
        w = self.testFaceMS @ self.U[:, :self.r]
        weights = []
        distances = []
        for ind, image in enumerate(self.imagesDatabaseArray):
            weight = image @ self.U[:, :self.r]
            dist = np.linalg.norm(weight-w)
            distances.append((dist, ind))
            # if dist<seuil :
            #     print(str(ind+1)+":"+str(dist))
            # if ind == 10 :
            # print("--> "+str(ind+1)+":"+str(dist))
            weights.append(weight)

        # dist = np.linalg.norm(weights[10]-weights[90])

        return distances

    # def findSeuil(self):
    #     w = self.testFaceMS @ self.U[:, :self.r]
    #     weights = []
    #     distances = []
    #     for ind, image in enumerate(self.imagesDatabaseArray):
    #         weight = image @ self.U[:, :self.r]
    #         dist = np.linalg.norm(weight-w)
    #         distances.append((dist, ind))
    #         weights.append(weight)

    #     dist2 = []
    #     res = 0
    #     i = 1
    #     k = 0
    #     for dist, ind in distances:
    #         if i < self.imagePerPerson+1:
    #             res += dist
    #             i += 1
    #         if i == self.imagePerPerson+1:
    #             res /= self.imagePerPerson
    #             dist2.append((res, k))
    #             k += 1
    #             res = 0
    #             i = 1

    #     return dist2

    def findSeuil2(self, distances):

        # w = self.testFaceMS @ self.U[:, :self.r]
        # weights = []
        # distances = []

        # listOfPersonImage = np.array_split(self.imagesDatabaseArray,19)

        # for ind,person in enumerate(listOfPersonImage):
        #     distance = []
        #     for image in person:
        #         weight = image @ self.U[:, :self.r]
        #         dist = np.linalg.norm(weight-w)
        #         distances.append(dist)
        #         #weightGroup.append(weight)

        #     distances.append((distance,ind))

        # #sum(1 if meets_condition(x) else 0 for x in my_list)

        # print(distances).
        threshold = 2700
        liste = {}
        for dist, ind in distances:
            # print(dist)
            if dist < threshold:
                # print(ind)
                if ind//self.imagePerPerson in self.acceptedListe:
                    self.acceptedListe[ind//self.imagePerPerson] += 1
                else:
                    self.acceptedListe[ind//self.imagePerPerson] = 1

        self.acceptedListe = {k: v for k, v in sorted(
            self.acceptedListe.items(), key=lambda item: item[1], reverse=True)}
        # print(self.acceptedListe)

        # return self.acceptedListe

    def printByName(self, liste):
        for dist, ind in liste:
            # print(ind)
            print(self.noms[ind]+"->"+str(dist))

    def printByName2(self, liste):
        for dist, ind in liste:

            print(self.noms[ind//self.imagePerPerson]+"->"+str(dist))

    def test1(self):
        ''' Si on a enlevé la n-1 personne on va la tester contre la base de donnée pour voir si elle est reconnu -> faux positif '''
        self.acceptedList = {}
        for i in range(1, self.imagePerPerson+1):
            self.fileNameAbsolute = self.pathDatabase + \
                format(self.personCount + 1, '02d')+'_'+format(i, '02d')+'.jpg'
            self.imageCropped = cv2.imread(self.fileNameAbsolute)
            if i == 1:
                self.displayImage(self.fileNameAbsolute,
                                  image_id_og, canvas_og)
            self.load(self.imageCropped)
            self.testImageR(1)

        print("Nombre de personnes : " + str(self.personCount) +
              " nb images par personnes : " + str(self.imagePerPerson))
        for ind, dist in self.acceptedListe.items():
            print(str(ind) + ': '+self.noms[ind]+"->"+str(dist))

        somme = sum(self.acceptedListe.values())
        total = self.personCount*self.imagePerPerson*self.imagePerPerson
        print(str(somme)+'/'+str(total))
        print("taux de faux positif : " + str(round(somme/total * 100, 2))+"%")

    def test2(self):
        ''' Si on a enlevé la n-1 image on va la tester contre la base de donnée pour voir si elle est reconnu -> vrai positif '''
        self.acceptedListe = {}
        for i in range(1, self.personCount+1):
            self.fileNameAbsolute = self.pathDatabase + \
                format(i, '02d')+'_' + \
                format(self.imagePerPerson + 1, '02d')+'.jpg'
            self.imageCropped = cv2.imread(self.fileNameAbsolute)
            if i == 1:
                self.displayImage(self.fileNameAbsolute,
                                  image_id_og, canvas_og)
            self.load(self.imageCropped)
            self.testImageR(1)

        print("Nombre de personnes : " + str(self.personCount) +
              " nb images par personnes : " + str(self.imagePerPerson))
        for ind, dist in self.acceptedListe.items():
            print(str(ind) + ': '+self.noms[ind]+"->"+str(dist))

        somme = sum(self.acceptedListe.values())
        total = self.personCount*self.imagePerPerson*self.imagePerPerson
        print(str(somme)+'/'+str(total))
        print("taux de vrai positif : " + str(round(somme/total * 100, 2))+"%")

    def testVPFP(self, idPerson, idImage, seuil):

        self.fileNameAbsolute = self.pathDatabase + \
            format(idPerson, '02d')+'_'+format(idImage, '02d')+'.jpg'
        self.imageCropped = cv2.imread(self.fileNameAbsolute)
        self.load(self.imageCropped)
        self.reconFace = self.averageImageDatabase + \
            self.U[:, :self.r]  @ self.U[:, :self.r].T @ self.testFaceMS
        distances = self.computeDistances()
        # distance = distances.sort()
        distance = [(a, b) for a, b in distances]

        VP = 0
        FP = 0
        VN = 0
        FN = 0
        for dist, ind in distance:
            if dist < seuil:
                if idImage == ind//self.imagePerPerson:
                    VP += 1
                else:
                    FP += 1
            else:
                if idImage == ind//self.imagePerPerson:
                    FN += 1
                else:
                    VN += 1

        return VP, FP, FN, VN

    def testVPFP2(self, idPerson, idImage, seuil):

        self.fileNameAbsolute = self.pathDatabase + \
            format(idPerson, '02d')+'_'+format(idImage, '02d')+'.jpg'
        self.imageCropped = cv2.imread(self.fileNameAbsolute)
        self.load(self.imageCropped)
        self.reconFace = self.averageImageDatabase + \
            self.U[:, :self.r]  @ self.U[:, :self.r].T @ self.testFaceMS
        distances = self.computeDistances()
        # distance = distances.sort()
        distance = [(a, b) for a, b in distances]

        VP = 0
        FP = 0
        VN = 0
        FN = 0
        for dist, ind in distance:
            if idImage == ind//self.imagePerPerson:
                if dist < seuil:
                    VP += 1
                else:
                    FP += 1
            else:
                if idImage == ind//self.imagePerPerson:
                    FN += 1
                else:
                    VN += 1

        return VP, FP, FN, VN

    def test3(self):
        # print(self.testVPFP(1,1,3500))
        sens = open("sens.dat", "w")
        spec = open("spec.dat", "w")

        final = open("average.dat", "w")

        # seuil = 5000
        totalImages = self.imagePerPerson * self.personCount
        comparedNumberImages = totalImages*self.imagePerPerson

        P = self.imagePerPerson
        N = (comparedNumberImages - self.imagePerPerson)

        print("P="+str(P))
        print("N="+str(N))
        for seuil in range(500, 8000, 1000):
            print("seuil="+str(seuil))
            scoreVP = []
            scoreFP = []
            scoreFN = []
            scoreVN = []
            for k in range(1, self.personCount+1):
                sumFP = 0
                sumVP = 0
                sumFN = 0
                sumVN = 0
                print("test de "+str(k))
                for i in range(1, self.imagePerPerson+1):
                    VP, FP, FN, VN = self.testVPFP(k, i, seuil)
                    sumVP += VP
                    sumFP += FP

                    sumFN += FN
                    sumVN += VN
                # print( (sumVP,sumFP))
                # scoreFP.append((noms[k],sumFP))
                scoreVP.append(sumVP)
                scoreFP.append(sumFP)

                scoreVN.append(sumVN)
                scoreFN.append(sumFN)

            VP = sum(i for i in scoreVP)
            FP = sum(i for i in scoreFP)

            VN = sum(i for i in scoreVN)
            FN = sum(i for i in scoreFN)
            print(VP)
            print(FP)
            print(FN)
            print(VN)

            FPR = FP/(totalImages*totalImages -
                      self.imagePerPerson*totalImages)
            sensi = VP/(totalImages*self.imagePerPerson)
            speci = 1-FPR
            TN = speci * N

            ACC = (VP + VN) / ((P + N)*self.personCount)
            print("accuracy : " + str(ACC))

            print("sensivity : " + str(sensi))
            print("specificity: "+str(speci))
            sens.write(str(seuil)+" "+str(sensi)+"\n")
            spec.write(str(seuil)+" "+str(speci)+"\n")

            final.write(str(seuil)+" "+str(ACC)+"\n")

    def test4(self):
        seuil = 1250
        for i in range(1, self.imagePerPerson+1):
            print(self.testVPFP2(self.personCount + 1, i, seuil))

    def test5(self):
        seuil = 1500
        for i in range(1, self.personCount+1):
            print(self.testVPFP2(i, self.imagePerPerson+1, seuil))

    # def computeSensSpec():
    #     self.r = 6
    #     P =5
    #     sens = open("sens.dat", "w")
    #     spec = open("spec.dat", "w")

    #     test2 = open("test.dat", "w")
    #     ind = 0

    #     for seuil in range(500,5000,500):
    #         scoreVP = []
    #         scoreFP = []
    #         print("seuil="+str(seuil))
    #         for k in range(self.personCount):
    #             sumFP = 0
    #             sumVP = 0
    #             #print("test de "+str(k)+"\n")
    #             for i in range( (self.imagePerPerson*k)+1,(self.imagePerPerson*k)+self.imagePerPerson+1):
    #                 VP,FP =testVPFP(i,r,k,seuil)
    #                 sumVP +=VP
    #                 sumFP +=FP
    #             #print( (sumVP,sumFP))
    #             #scoreFP.append((noms[k],sumFP))
    #             scoreVP.append(sumVP)
    #             scoreFP.append(sumFP)

    #         VP =sum(i for i in scoreVP )
    #         FP =sum(i for i in scoreFP )

    #         FPR = FP/N
    #         sensi =VP/P
    #         speci = 1-FPR
    #         print("sensivity : " + str(sensi))
    #         print("specificity: "+str(speci))
    #         sens.write(str(seuil)+" "+str(sensi)+"\n")
    #         spec.write(str(seuil)+" "+str(speci)+"\n")

    #         test2.write(str(seuil)+" "+str(speci+sensi)+"\n")

    def compute_and_display_ressemblance(self, other):
        # Met à jour le seuil
        self.seuil = int(self.spinBoxSeuil.get())
        distances = self.computeDistances()
        distances.sort()
        if other == 0:
            print("distance individuelle : ")
            self.printByName2(distances[:5])

        # print("\n Nombre d'image par personne reconnue sous le seuil : ")
        self.findSeuil2(distances)
        # print(self.acceptedListe)

        # distanceGroup.sort()
        # self.printByName(distanceGroup[:5])

        # print("\ndistance GRoupe : ")
        # distanceGroup = self.findSeuil()
        # distanceGroup.sort()
        # self.printByName(distanceGroup[:5])

        dist, ind = distances[0]
        # name = self.noms[ind//5]

        name = self.noms[ind//self.imagePerPerson]

        if (dist < self.seuil):
            label_ressemblance.configure(
                text="Identification : " + name + "(" + str(int(dist)) + ")", fg="#20b02c")
        else:
            label_ressemblance.configure(
                text="Inconnu : " + "(" + name + " " + str(int(dist)) + ")", fg="#b02020")

    def testImageR(self, autre):

        # if self.slider != None:
        #     self.r = self.slider.get()
        # else:
        #     self.r = 8
        # print(self.spinBoxDimension.get())
        if self.spinBoxDimension != None:
            self.r = int(self.spinBoxDimension.get())
        else:
            self.r = 8

        # print(self.r)

        # display(self.average)
        self.reconFace = self.averageImageDatabase + \
            self.U[:, :self.r]  @ self.U[:, :self.r].T @ self.testFaceMS

        recons = np.reshape(
            self.reconFace, (self.croppedResolution, self.croppedResolution))

        # display(recons)

        # ---------------------------------------------------------------------------------------------
        # Ouvre l'image reconstruite

        # create the image object, and save it so that it
        # won't get deleted by the garbage collector
        canvas_rec.image_tk = ImageTk.PhotoImage(
            Image.fromarray(recons).resize((300, 300), Image.Resampling.LANCZOS))
        # configure the canvas item to use this image
        canvas_rec.itemconfigure(image_id_rec, image=canvas_rec.image_tk)
        # ---------------------------------------------------------------------------------------------
        # testFaceMS = test(1)
        # display(testFaceMS)

        self.compute_and_display_ressemblance(autre)

    def show_frames(self, i, une_frame_sur):
        """Define function to show frame"""

        # Capture frame-by-frame
        if self.pauseVideoFlag == False:
            i += 1
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            basewidth = 400
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label_webcam.imgtk = imgtk
            label_webcam.configure(image=imgtk)

            if i == une_frame_sur:
                # self.imageWebcam = img
                self.imageWebcam = cv2image
                # self.crop_webcam()
                self.crop_webcam()
                self.displayImage(self.fileNameAbsolute,
                                  image_id_og, canvas_og)

                self.load(self.imageCropped)
                self.compute_and_display_ressemblance(other=0)

                i = 0
            # Repeat after an interval to capture continiously
        label_webcam.after(20, lambda: self.show_frames(i, une_frame_sur))

    def show_frames_aligned(self, i, une_frame_sur):
        """Define function to show frame"""

        # Capture frame-by-frame
        if self.pauseVideoFlag == False:
            i += 1
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            basewidth = 400
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label_webcam.imgtk = imgtk
            label_webcam.configure(image=imgtk)

            if i == une_frame_sur:
                # self.imageWebcam = img
                self.imageWebcam = cv2image
                success = self.crop_webcam_aligned()
                if success:
                    self.displayImage(self.fileNameAbsolute,
                                      image_id_og, canvas_og)

                    self.load(self.imageCropped)
                    self.compute_and_display_ressemblance(other=0)

                i = 0
            # Repeat after an interval to capture continiously
        label_webcam.after(20, lambda: self.show_frames(i, une_frame_sur))

    def show_frames_phone(self, i, une_frame_sur):
        """Define function to show frame from phone"""

        # Capture frame-by-frame
        if self.pauseVideoFlag == False:
            i += 1
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap_phone.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            basewidth = 400
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label_webcam.imgtk = imgtk
            label_webcam.configure(image=imgtk)

            if i == une_frame_sur:
                # self.imageWebcam = img
                self.imageWebcam = cv2image
                self.crop_webcam()
                self.displayImage(self.fileNameAbsolute,
                                  image_id_og, canvas_og)

                self.load(self.imageCropped)
                self.compute_and_display_ressemblance(other=0)

                i = 0
            # Repeat after an interval to capture continiously
        label_webcam.after(
            20, lambda: self.show_frames_phone(i, une_frame_sur))

    def show_frames_phone_aligned(self, i, une_frame_sur):
        """Define function to show frame from phone"""

        # Capture frame-by-frame
        if self.pauseVideoFlag == False:
            i += 1
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(cap_phone.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            basewidth = 400
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label_webcam.imgtk = imgtk
            label_webcam.configure(image=imgtk)

            if i == une_frame_sur:
                # self.imageWebcam = img
                self.imageWebcam = cv2image
                success = self.crop_webcam_aligned()
                if success:
                    self.displayImage(self.fileNameAbsolute,
                                      image_id_og, canvas_og)

                    self.load(self.imageCropped)
                    self.compute_and_display_ressemblance(other=0)

                i = 0
            # Repeat after an interval to capture continiously
        label_webcam.after(
            20, lambda: self.show_frames_phone_aligned(i, une_frame_sur))
    #######################################################################################################################################################################

    def show_spinboxes(self):
        self.frameSpinbox = Frame(window)
        self.labelDimension = Label(self.frameSpinbox, font=(
            "Arial", 18), fg='Black', text="Dimension :")
        self.labelDimension.grid(row=0, column=0, sticky=W, padx=10)
        self.currentValueDimension = StringVar()
        self.currentValueDimension.set("8")
        self.spinBoxDimension = ttk.Spinbox(
            self.frameSpinbox,
            from_=0,
            to=self.imagesCount,
            textvariable=self.currentValueDimension,
            command=lambda: self.testImageR(0),
            width=5)

        self.spinBoxDimension.grid(row=0, column=1, sticky=W, padx=10)
        self.onlyOneSlider = True

        self.labelSeuil = Label(self.frameSpinbox, font=(
            "Arial", 18), fg='Black', text="Seuil :")
        self.labelSeuil.grid(row=0, column=4, sticky=W, padx=10)
        self.currentValueSeuil = StringVar()
        # self.currentValueSeuil.set(str(int(self.theta)))
        self.currentValueSeuil.set("3800")

        self.spinBoxSeuil = ttk.Spinbox(
            self.frameSpinbox,
            from_=0,
            to=100000,
            textvariable=self.currentValueSeuil,
            command=lambda: self.compute_and_display_ressemblance(0),
            width=5)

        self.spinBoxSeuil.grid(row=0, column=5, sticky=W, padx=10)
        self.frameSpinbox.pack(expand=YES)

        self.onlyOneSlider = True

    def open_file(self):
        """Affiche la fenêtre d'ouverture de fichiers"""
        print("Open file")
        self.fileNameAbsolute = fd.askopenfilename(title="Choisis ton image")
        cwd = os.getcwd()
        # remplace absolu en relatif
        self.fileNameRelative = self.fileNameAbsolute.replace(
            format(cwd)+'/', "")
        print("Opened " + self.fileNameRelative)

        ############################################################################
        if(self.fileNameAbsolute):
            self.crop_filepath(self.fileNameAbsolute)
            # self.display()
            self.displayImage(self.fileNameAbsolute, image_id_og, canvas_og)
            # self.displayImage(self.fileNameAbsolute, image_id_rec, canvas_rec)

            self.load(self.imageCropped)

            # Création slider
            if self.onlyOneSlider == False:
                self.show_spinboxes()

            self.testImageR(autre=0)

    def open_webcam(self):
        if self.onlyOneSlider == False:
            self.show_spinboxes()

        # self.show_frames(0, 20)
        self.show_frames_aligned(0, 20)

    def open_phonecam(self):
        if self.onlyOneSlider == False:
            self.show_spinboxes()

        # self.show_frames_phone(0, 20)
        self.show_frames_phone_aligned(0, 20)


if __name__ == '__main__':

    # path = 'ressource/dataset/database5/face'
    # croppedResolution = 64
    # nImage = 5
    # equalizedAndBlur = False
    # noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "erwan", "ange", "roland", "aurelien",
    #         "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]

    print("Choix de la base de donnée : ")
    print("1. Originale")
    print("2. Visages Alignés")
    print("3. Visages Alignés puis égalisés puis flouttés")
    print("4. Visages Alignés puis égalisés puis flouttés puis augmentés avec spécification d'histogramme")
    choix = input()
    choix = int(choix)
    if choix == 1:
        path = 'ressource/dataset/database25Enhanced/face'
        croppedResolution = 64
        nImage = 25
        equalizedAndBlur = False
        noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "erwan", "ange", "roland", "aurelien",
                "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]
    if choix == 2:
        path = 'ressource/dataset/new/aligned/face'
        croppedResolution = 64
        nImage = 25
        equalizedAndBlur = False
        noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "erwan", "ange", "roland", "aurelien",
                "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]
    if choix == 3:
        path = 'ressource/dataset/new/aligned_equalized_blurred/face'
        croppedResolution = 64
        nImage = 25
        equalizedAndBlur = True
        noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "erwan", "ange", "roland", "aurelien",
                "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]
    if choix == 4:
        path = 'ressource/dataset/new/aligned_equalized_blurred_spec/face'
        croppedResolution = 64
        nImage = 89
        equalizedAndBlur = True
        noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "erwan", "ange", "roland", "aurelien",
                "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]

    nPerson = 19

    reconnaissance = Reconnaissance(
        path, croppedResolution, noms, nImage, nPerson, nPerson*nImage, equalizedAndBlur)

    # cameraFeedURL = "https://192.168.1.23:8080/video"
    cameraFeedURL = "https://192.168.1.10:8080/video"

    window = Tk()

    window.title("Reconnaissance Faciale")
    window.geometry("720x800")
    window.minsize(352, 240)
    # window.iconbitmap("facial-recognition.ico")
    # window.config(background='#4065A4')

    # Creer la frame principale
    frame = Frame(window)

    # Création d'image
    width = 300
    height = 300

    # Image à reconnaître
    canvas_og = Canvas(frame, width=width, height=height,
                       bd=0, highlightthickness=0)
    # canvas.create_image(width/2, height/2, image=image_base)
    image_id_og = canvas_og.create_image(0, 0, anchor="nw")
    canvas_og.grid(row=0, column=0, sticky=W, padx=10)

    # Image reconstruite
    canvas_rec = Canvas(frame, width=width, height=height,
                        bd=0, highlightthickness=0)
    image_id_rec = canvas_rec.create_image(0, 0, anchor="nw")
    canvas_rec.grid(row=0, column=1, sticky=W, padx=10)

    # Afficher la frame
    frame.pack(expand=YES)

    # Create a Label to capture the Video frames
    label_webcam = Label(window)
    label_webcam.pack(pady=10)
    cap = cv2.VideoCapture(0)
    cap_phone = cv2.VideoCapture(cameraFeedURL)

    # Affichage Texte Ressemblance
    label_ressemblance = Label(window, font=("Arial", 18), fg='Black')
    label_ressemblance.pack(pady=10)

    # Création d'une barre de menu
    menu_bar = Menu(window)

    # Créer un premier menu
    file_menu = Menu(menu_bar, tearoff=0)
    file_menu.add_command(
        label="Ouvrir une Image", command=reconnaissance.open_file, accelerator="Ctrl+O")
    file_menu.add_command(
        label="Ouvrir flux vidéo de la webcam", command=reconnaissance.open_webcam, accelerator="Ctrl+W")
    file_menu.add_command(
        label="Ouvrir flux vidéo à partir du téléphone", command=reconnaissance.open_phonecam, accelerator="Ctrl+X")
    file_menu.add_command(
        label="Quitter", command=window.quit, accelerator="Ctrl+Q")

    test_menu = Menu(menu_bar, tearoff=0)
    test_menu.add_command(
        label="Test1", command=reconnaissance.test1, accelerator="Ctrl+1")
    test_menu.add_command(
        label="Test2", command=reconnaissance.test2, accelerator="Ctrl+2")
    test_menu.add_command(
        label="Test3", command=reconnaissance.test3, accelerator="Ctrl+3")
    test_menu.add_command(
        label="Test4", command=reconnaissance.test4, accelerator="Ctrl+4")
    test_menu.add_command(
        label="Test5", command=reconnaissance.test5, accelerator="Ctrl+5")
    test_menu.add_command(
        label="Quitter", command=window.quit, accelerator="Ctrl+Q")
    menu_bar.add_cascade(label="Fichier", menu=file_menu)
    menu_bar.add_cascade(label="Tests", menu=test_menu)

    # Configurer notre fenêtre pour ajouter cette men bar
    window.config(menu=menu_bar)

    # Shortcuts
    window.bind_all("<Control-o>", lambda e: reconnaissance.open_file())
    window.bind_all("<Control-w>", lambda e: reconnaissance.open_webcam())
    window.bind_all("<Control-x>", lambda e: reconnaissance.open_phonecam())

    window.bind_all("<p>", lambda e: reconnaissance.switch_pauseFlag())
    window.bind_all("<space>", lambda e: reconnaissance.switch_pauseFlag())

    window.bind_all("<Control-q>", lambda e: window.quit())

    # Afficher la fenêtre
    window.mainloop()
