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
    def __init__(self,path, n):
        self.pathDatabase = path
        self.fileNameAbsolute = ""
        self.fileNameRelative = ""
        self.imagesDatabase = []
        self.imagesCount = n
        self.imagesDatabaseArray = None
        self.averageImageDatabase = None
        self.imageCropped = None

        self.loadImages()
        self.preProcess()

    def loadImages(self):
        """On charge notre dataset des étudiants d'imagine en gardant les (n-1) premiers donc 95 images sur 100 images. (Il y a 5 images par étudiant)"""
        print("Laoding database ..")

        for i in range(1,self.imagesCount+1):
            stri = self.pathDatabase+format(i, '03d')+'.jpg'
            img =cv2.imread(stri)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.ndarray.flatten(gray)
            self.imagesDatabase.append(gray)
        print("Database loaded !")

    def preProcess(self):
        """On transforme notre tableau d'image en numpy array et on enlève la moyenne des visage afin de centrer notre nuage de point et de pouvoir calculer une ACP. """
        self.imagesDatabaseArray = np.array(self.imagesDatabase)
        self.averageImageDatabase = np.mean(self.imagesDatabaseArray,axis=0)
        self.imagesDatabaseArray = self.imagesDatabaseArray - np.tile(self.averageImageDatabase,(self.imagesDatabaseArray.shape[0],1))

    def crop(self):
        """On charge l'image du filepath et on la crop et on l'écrit en dur sur le disque """

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.imageCropped = cv2.imread(self.self.fileNameAbsolute) 
        gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        hmax =-1
        indice = -1
        ind = 0
        offset =0
        for face in faces:
            if (face[2]+face[3]) > hmax :
                hmax = face[2]+face[3]
                indice = ind
            ind+=(1)

        x= faces[indice][0]
        y =faces[indice][1]
        w=faces [indice][2]
        h=faces [indice][3]
        self.imageCropped = self.imageCropped[y:y+h, x:x+w]
        self.imageCropped = cv2.resize(self.imageCropped, (64,64), interpolation = cv2.INTER_AREA)
        cv2.imwrite("cropped.jpg", self.imageCropped)

    def open_file(self):
        """Affiche la fenêtre d'ouverture de fichiers"""
        print("Open file")
        self.fileNameAbsolute = fd.askopenfilename(title="Choisis ton image")
        cwd = os.getcwd()
        # remplace absolu en relatif
        self.fileNameRelative = filename.replace(format(cwd)+'/', "")
        print("Openned "+ self.fileNameRelative)
        self.crop()
        self.display()
    def displayImage(self, canvas):
        if(self.fileNameAbsolute):
            # Ouvre l'image
            image = Image.open(self.fileNameAbsolute)
            w, h = image.size
            
            gray = cv2.cvtColor(self.imageCropped, cv2.COLOR_BGR2GRAY)
            # create the image object, and save it so that it
            # won't get deleted by the garbage collector
            canvas.image_tk = ImageTk.PhotoImage(Image.fromarray(gray).resize((300, 300), Image.ANTIALIAS))
            # configure the canvas item to use this image
            canvas.itemconfigure(image_id, image=canvas.image_tk)
         

if __name__ == '__main__':
    path = 'ressource/dataset/croppedfaces64/face'
    n = 90
    reconnaissance = Reconnaissance(path,n)

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
    canvas_og = Canvas(frame, width=width, height=height, bd=0, highlightthickness=0)
    # canvas.create_image(width/2, height/2, image=image_base)
    image_id_og = canvas_og.create_image(0, 0, anchor="nw")
    canvas_og.grid(row=0 , column=0 , sticky=W, padx=10)

    # Image reconstruite 
    canvas_rec = Canvas(frame, width=width, height=height, bd=0, highlightthickness=0)
    image_id_rec = canvas_rec.create_image(0, 0, anchor="nw")
    canvas_rec.grid(row=0 , column=1 , sticky=W, padx=10)





    # Afficher la frame
    frame.pack(expand=YES)
    # Affichage Texte Ressemblance
    label_title = Label(window, font=("Arial", 18), fg='Black')
    label_title.pack(pady=25)
    # Création d'une barre de menu
    menu_bar = Menu(window)

    # Créer un premier menu
    file_menu = Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Ouvrir", command=reconnaissance.open_file, accelerator="Ctrl+O")
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