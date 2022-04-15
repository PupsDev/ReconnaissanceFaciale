import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import cv2
import sys

from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk

from functools import partial


class Reconnaissance:
    def __init__(self, path, noms, n):
        self.pathDatabase = path
        self.fileNameAbsolute = ""
        self.fileNameRelative = ""
        self.imagesCount = n
        self.noms = noms
        self.imagesDatabase = []
        self.imagesDatabaseArray = None
        self.averageImageDatabase = None
        self.imageCropped = None
        self.reconFace = None

        self.transpose = None
        self.U = None
        self.S = None
        self.VT = None
        self.r = 50
        self.seuil = 2700

        self.slider = None

        # self.loadImages()
        # self.preProcess()
        # self.load()

    def crop(self):
        """On charge l'image du filepath et on la crop et on l'écrit en dur sur le disque """

        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.imageCropped = cv2.imread(self.fileNameAbsolute)
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        hmax = -1
        indice = -1
        ind = 0
        offset = 0
        for face in faces:
            if (face[2]+face[3]) > hmax:
                hmax = face[2]+face[3]
                indice = ind
            ind += (1)

        x = faces[indice][0]
        y = faces[indice][1]
        w = faces[indice][2]
        h = faces[indice][3]
        self.imageCropped = self.imageCropped[y:y+h, x:x+w]
        self.imageCropped = cv2.resize(
            self.imageCropped, (64, 64), interpolation=cv2.INTER_AREA)
        cv2.imwrite("cropped.jpg", self.imageCropped)

    def displayImage(self, fileNameAbsolute, image_id, canvas):
        """Affiche l'image dans le canvas"""
        if(self.fileNameAbsolute):
            # Ouvre l'image
            image = Image.open(fileNameAbsolute)
            w, h = image.size

            gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
            # create the image object, and save it so that it
            # won't get deleted by the garbage collector
            canvas.image_tk = ImageTk.PhotoImage(
                Image.fromarray(gray).resize((300, 300), Image.ANTIALIAS))
            # configure the canvas item to use this image
            canvas.itemconfigure(image_id, image=canvas.image_tk)

    def loadImages(self):
        """On charge notre dataset des étudiants d'imagine en gardant les (n-1) premiers donc 95 images sur 100 images. (Il y a 5 images par étudiant)"""
        print("Laoding database ..")

        for i in range(1, self.imagesCount+1):
            stri = self.pathDatabase+format(i, '03d')+'.jpg'
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

    def load(self):
        # img = cv2.imread(filepath)
        # img = self.crop(filepath)
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)

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
            #print("--> "+str(ind+1)+":"+str(dist))
            weights.append(weight)

        #dist = np.linalg.norm(weights[10]-weights[90])

        return distances

    def findSeuil(self):
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
            #print("--> "+str(ind+1)+":"+str(dist))
            weights.append(weight)

        dist2 = []
        res = 0
        i = 1
        k = 0
        for dist, ind in distances:
            if i < 6:
                res += dist
                i += 1
            if i == 6:
                res /= 5
                dist2.append((res, k))
                k += 1
                res = 0
                i = 1

        return dist2

    def printByName(self, liste):
        for dist, ind in liste:
            # print(ind)
            print(self.noms[ind]+"->"+str(dist))

    def printByName2(self, liste):
        for dist, ind in liste:
            # print(ind)
            print(self.noms[ind//5]+"->"+str(dist))

    def testImageR(self, autre):
        self.r = self.slider.get()
        print(self.r)

        # display(self.average)
        self.reconFace = self.averageImageDatabase + \
            self.U[:, :self.r]  @ self.U[:, :self.r].T @ self.testFaceMS

        recons = np.reshape(self.reconFace, (64, 64))

        # display(recons)

        # ---------------------------------------------------------------------------------------------
        # Ouvre l'image reconstruite

        # create the image object, and save it so that it
        # won't get deleted by the garbage collector
        canvas_rec.image_tk = ImageTk.PhotoImage(
            Image.fromarray(recons).resize((300, 300), Image.ANTIALIAS))
        # configure the canvas item to use this image
        canvas_rec.itemconfigure(image_id_rec, image=canvas_rec.image_tk)
        # ---------------------------------------------------------------------------------------------
        #testFaceMS = test(1)
        # display(testFaceMS)

        distances = self.computeDistances()
        distances.sort()
        print("distance individuelle : ")
        distances = self.computeDistances()
        distances.sort()
        self.printByName2(distances[:5])

        print("\ndistance GRoupe : ")
        distanceGroup = self.findSeuil()
        distanceGroup.sort()
        self.printByName(distanceGroup[:5])

        dist, ind = distances[0]
        name = self.noms[ind//5]
        label_ressemblance.configure(
            text="Ressemblance : " + name + " distance : " + str(dist))

    #######################################################################################################################################################################

    def open_file(self):
        """Affiche la fenêtre d'ouverture de fichiers"""
        print("Open file")
        self.fileNameAbsolute = fd.askopenfilename(title="Choisis ton image")
        cwd = os.getcwd()
        # remplace absolu en relatif
        self.fileNameRelative = self.fileNameAbsolute.replace(
            format(cwd)+'/', "")
        print("Openned " + self.fileNameRelative)

        ############################################################################
        if(self.fileNameAbsolute):
            self.crop()
            # self.display()
            self.displayImage(self.fileNameAbsolute, image_id_og, canvas_og)
            # self.displayImage(self.fileNameAbsolute, image_id_rec, canvas_rec)
            self.loadImages()
            self.preProcess()
            self.transpose = self.imagesDatabaseArray.T
            self.U, self.S, self.VT = np.linalg.svd(
                self.transpose, full_matrices=0)
            self.load()

            # Création slider
            self.slider = Scale(window, from_=1, to=self.r, orient=HORIZONTAL,
                                command=self.testImageR)

            self.slider.pack(pady=10)
            # r = slider.get()
            # print(r)


if __name__ == '__main__':
    path = 'ressource/dataset/croppedfaces64/face'
    n = 90
    noms = ["gauthier", "albena", "mathieu", "alexandre F", "dorian", "thomas ?", "erwan", "ange", "roland", "aurelien",
            "samuel", "alexandre S", "florentin", "sylvain", "khélian", "camille", "marius", "alexandre L", "thomas S", "maxime"]
    reconnaissance = Reconnaissance(path, noms, n)

    window = Tk()

    window.title("Reconnaissance Faciale")
    window.geometry("720x480")
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

    # Affichage Texte Ressemblance
    label_ressemblance = Label(window, font=("Arial", 18), fg='Black')
    label_ressemblance.pack(pady=25)

    # Création d'une barre de menu
    menu_bar = Menu(window)

    # Créer un premier menu
    file_menu = Menu(menu_bar, tearoff=0)
    file_menu.add_command(
        label="Ouvrir", command=reconnaissance.open_file, accelerator="Ctrl+O")
    file_menu.add_command(
        label="Quitter", command=window.quit, accelerator="Ctrl+Q")

    menu_bar.add_cascade(label="Fichier", menu=file_menu)

    # Configurer notre fenêtre pour ajouter cette men bar
    window.config(menu=menu_bar)

    # Shortcuts
    window.bind_all("<Control-o>", lambda e: reconnaissance.open_file())
    window.bind_all("<Control-q>", lambda e: window.quit())

    # Afficher la fenêtre
    window.mainloop()
