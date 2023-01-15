#Import package
import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract

def create_list_image():
    list_files = []
    all_images = os.listdir("./Labelisation/Images")
    for image in all_images:#Je crée une boucle for pour changer tous mes fichiers pdf en images
            pil_image = np.array(image)  #
            opencvImage = cv2.cvtColor(np.array(pil_image),cv2.COLOR_BGR2GRAY)  # Je Convertie l'image en gris à l'aide de la fonction CvtColor
            list_files.append(image)

    all_images = os.listdir("./Labelisation/Textes")
    for image in all_images:  # Je crée une boucle for pour changer tous mes fichiers pdf en images
        pil_image = np.array(image)  #
        opencvImage = cv2.cvtColor(np.array(pil_image),
                                   cv2.COLOR_BGR2GRAY)  # Je Convertie l'image en gris à l'aide de la fonction CvtColor
        list_files.append(image)
    return list_files

def convert_to_text(files_of_image):
    #Appliquer OCR (Optical Character Recognition) sur les images pour extraire les textes
    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\baptiste\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
    #On utilise le package pytesseract pour transformer une image en texte
    list_text = []
    for docu in files_of_image:#Je crée une  boucle for pour permettre aux images de donner leur texte

        custom_config = r'--oem 3 --psm 6'
        result_txt = pytesseract.image_to_string(docu, config=custom_config, lang="eng")
        list_text.append(result_txt)#Ajout des texte à la liste des textes.
    return list_text

def create_dictionnary_of_text(id_start,files_of_text):
    dico_carac = {}
    for carac in files_of_text:
        a = len(carac)
        dico_carac[id_start] = a
        id_start = id_start + 1
    return dico_carac

#  link_poppler = "C:/Users/baptiste/Documents/Etude/CNAM/Année2/Python/poppler-0.68.0/bin"

print(create_list_image())
files = create_list_image()

list_text = convert_to_text(files)

id = 1
pdf_dictionnary = create_dictionnary_of_text(id,list_text)
print(pdf_dictionnary)