import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import time


# All variables import
import os
import glob
from PIL import Image
from pdf2image import convert_from_path
from collections import Counter
import numpy as np
import scipy.spatial as sp
import cv2
import csv
from skimage import io
from pytesseract import pytesseract
import time
import shutil

# All Variables

def delete_temp_files():
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    files = glob.glob(path_to_project+'/1. Data/temp images/*')
    for f in files:
        os.remove(f)

    files = glob.glob(path_to_project+'/1. Data/temp pixelated images/*')
    for f in files:
        os.remove(f)

    files = glob.glob(path_to_project+'/1. Data/temp PDF/*')
    for f in files:
        os.remove(f)

def delete_flask_files():
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    files = glob.glob(path_to_project+'/Flask App/static/image/*')
    for f in files:
        os.remove(f)

    files = glob.glob(path_to_project+'/Flask App/static/texte/*')
    for f in files:
        os.remove(f)

def create_images(image, file):
    """Create a jpeg in the directory './Labelisation'"""
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    filename = file.replace('.pdf', '')
    for i in range(len(image)):
        if len(image) > 1:
            new_filename = path_to_project+'/1. Data/temp images/' + filename + '-' + str(i) + '.jpeg'
        else:
            new_filename = path_to_project+'/1. Data/temp images/' + filename + '.jpeg'
        image[i].save(new_filename, 'JPEG') 

def create_all_images():
    """Creates images in the directory images from the PDFs"""
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"
    files = os.listdir(path_to_project+"/1. Data/temp PDF")
    print(files)
    for file in files:
        image = convert_from_path(path_to_project+"/1. Data/temp PDF/" + file, 500, use_pdftocairo=True, strict=False)
        create_images(image, file)

def create_pixel_image(imagePath, newDirectory):
    img = Image.open(imagePath)
    A4_proportion = 297 / 210
    n = 35 # Number of pixel in the larger of the page
    imgSmall = img.resize((n, int(n * A4_proportion)), resample=Image.Resampling.BILINEAR)
    imageName = imagePath.split('/')[-1]
    newPath = newDirectory + imageName
    imgSmall.save(newPath)

def pixelize_images():
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    image_names = os.listdir(path_to_project+"/1. Data/temp images/")
    for image in image_names:
        path = path_to_project+'/1. Data/temp images/'+image
        create_pixel_image(path,path_to_project+"/1. Data/temp pixelated images/")

def get_colors(imagePath):
    main_colors = [(128, 0, 0),
               (139, 0, 0),
               (165, 42, 42),
               (178, 34, 34),
               (220, 20, 60),
               (255, 0, 0),
               (255, 99, 71),
               (255, 127, 80),
               (205, 92, 92),
               (240, 128, 128),
               (233, 150, 122),
               (250, 128, 114),
               (255, 160, 122),
               (255, 69, 0),
               (255, 140, 0),
               (255, 165, 0),
               (255, 215, 0),
               (184, 134, 11),
               (218, 165, 32),
               (238, 232, 170),
               (189, 183, 107),
               (240, 230, 140),
               (128, 128, 0),
               (255, 255, 0),
               (154, 205, 50),
               (85, 107, 47),
               (107, 142, 35),
               (124, 252, 0),
               (127, 255, 0),
               (173, 255, 47),
               (0, 100, 0),
               (0, 128, 0),
               (34, 139, 34),
               (0, 255, 0),
               (50, 205, 50),
               (144, 238, 144),
               (152, 251, 152),
               (143, 188, 143),
               (0, 250, 154),
               (0, 255, 127),
               (46, 139, 87),
               (102, 205, 170),
               (60, 179, 113),
               (32, 178, 170),
               (47, 79, 79),
               (0, 128, 128),
               (0, 139, 139),
               (0, 255, 255),
               (0, 255, 255),
               (224, 255, 255),
               (0, 206, 209),
               (64, 224, 208),
               (72, 209, 204),
               (175, 238, 238),
               (127, 255, 212),
               (176, 224, 230),
               (95, 158, 160),
               (70, 130, 180),
               (100, 149, 237),
               (0, 191, 255),
               (30, 144, 255),
               (173, 216, 230),
               (135, 206, 235),
               (135, 206, 250),
               (25, 25, 112),
               (0, 0, 128),
               (0, 0, 139),
               (0, 0, 205),
               (0, 0, 255),
               (65, 105, 225),
               (138, 43, 226),
               (75, 0, 130),
               (72, 61, 139),
               (106, 90, 205),
               (123, 104, 238),
               (147, 112, 219),
               (139, 0, 139),
               (148, 0, 211),
               (153, 50, 204),
               (186, 85, 211),
               (128, 0, 128),
               (216, 191, 216),
               (221, 160, 221),
               (238, 130, 238),
               (255, 0, 255),
               (218, 112, 214),
               (199, 21, 133),
               (219, 112, 147),
               (255, 20, 147),
               (255, 105, 180),
               (255, 182, 193),
               (255, 192, 203),
               (250, 235, 215),
               (245, 245, 220),
               (255, 228, 196),
               (255, 235, 205),
               (245, 222, 179),
               (255, 248, 220),
               (255, 250, 205),
               (250, 250, 210),
               (255, 255, 224),
               (139, 69, 19),
               (160, 82, 45),
               (210, 105, 30),
               (205, 133, 63),
               (244, 164, 96),
               (222, 184, 135),
               (210, 180, 140),
               (188, 143, 143),
               (255, 228, 181),
               (255, 222, 173),
               (255, 218, 185),
               (255, 228, 225),
               (255, 240, 245),
               (250, 240, 230),
               (253, 245, 230),
               (255, 239, 213),
               (255, 245, 238),
               (245, 255, 250),
               (112, 128, 144),
               (119, 136, 153),
               (176, 196, 222),
               (230, 230, 250),
               (255, 250, 240),
               (240, 248, 255),
               (248, 248, 255),
               (240, 255, 240),
               (255, 255, 240),
               (240, 255, 255),
               (255, 250, 250),
               (0, 0, 0),
               (105, 105, 105),
               (128, 128, 128),
               (169, 169, 169),
               (192, 192, 192),
               (211, 211, 211),
               (220, 220, 220),
               (245, 245, 245),
               (255, 255, 255)
               ]
    image = io.imread(imagePath)
    pixels = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB image
    h, w, bpp = np.shape(image)
    for py in range(0, h):
        for px in range(0, w):
            # Find the nearest color
            input_color = (image[py][px][0], image[py][px][1], image[py][px][2])
            tree = sp.KDTree(main_colors)
            ditsance, result = tree.query(input_color)
            nearest_color = main_colors[result]

            if nearest_color[0] != nearest_color[1] or nearest_color[1] != nearest_color[2] or nearest_color[0] != \
                    nearest_color[2]:
                pixels.append((nearest_color[0], nearest_color[1], nearest_color[2]))

    all_pixels = Counter(pixels)
    ColorNumber = len(all_pixels)
    MostFrequentColor = max(all_pixels, key=all_pixels.get)
    SecondMostFrequentColor = sorted(all_pixels, key=all_pixels.get)[-2]

    imageName = imagePath.split('/')[-1]

    dico_couleur_1 = {}
    dico_couleur_2 = {}
    for couleur in main_colors:
        dico_couleur_1[couleur] = 0
        dico_couleur_2[couleur] = 0
    
    dico_couleur_1[MostFrequentColor] = 1
    dico_couleur_2[SecondMostFrequentColor] = 1

    # normalize
    ColorNumber = (ColorNumber - 1) / (1+78)
    my_list = list(dico_couleur_1.values()) + list(dico_couleur_2.values()) 
    return [imageName, ColorNumber] + my_list
    
def create_dictionnary_of_proportion(files_of_image,dico_area_image,dico_area):
    dico_proportion = {}
    for imagei in files_of_image:
        print(dico_area[imagei])
        print(dico_area_image[imagei])
        dico_proportion[imagei] = int(dico_area[imagei])/int(dico_area_image[imagei])
    return dico_proportion

def create_characters_values(files_of_image):
    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    dico_images = {}
    for imagei in files_of_image:  
        path = path_to_project+"/1. Data/temp images/" + str(imagei)
        img = cv2.imread(path, 0)
        
        result_txt = pytesseract.image_to_string(img)
        a = len(result_txt)

        # normalize
        a= (a-1)/(1+8489)

        dico_images[imagei] = [a] # nb chars    
        
        x,y = img.shape        
        shape = x*y        

        # normalize
        shape = (shape - 11687500)/(11687500+24177345)
        dico_images[imagei].append(shape) # area

        letter_somme = 0        
        letter_boxes = pytesseract.image_to_boxes(img)
        # dico_images[imagei].append(len(letter_boxes.splitlines()))
        height, width = img.shape        
        for box in letter_boxes.splitlines():
            box = box.split()
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            cv2.rectangle(img, (x, height - y), (w, height - h), (0, 0, 255))
            cv2.putText(img, box[0], (x, height - h + 32), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            letter_somme = letter_somme + (w-x)*(h-y)

        # normalize
        letter_somme = letter_somme/25599697
        dico_images[imagei].append(letter_somme)

        dico_images[imagei].append(round(letter_somme/shape,2))

    return dico_images


def forward_propagation(X, parametres): 
    activations = {'A0' : X} 
    nbCouche = len(parametres) // 2 
    for couche in range(1, nbCouche+1): 
        Z = parametres['W' + str(couche)].dot(activations['A' + str(couche-1)]) + parametres['b' + str(couche)] 
        activations['A' + str(couche)] = 1 / (1+np.exp(-Z)) 
    return activations

def predict(X): 
    parametres = {'W1': np.array([[ 1.76, 0.4 , 0.98, 2.24, 1.87, -0.98, 0.95, -0.15, -0.1 , 0.41, 0.14, 1.45, 0.76, 0.12, 0.44, 0.33, 1.49, -0.21, 0.31, -0.85, -2.55, 0.65, 0.86, -0.74, 2.27, -1.45, 0.05, -0.19, 1.53, 1.47, 0.15, 0.38, -0.89, -1.98, -0.35, 0.16, 1.23, 1.2 , -0.39, -0.3 , -1.05, -1.42, -1.71, 1.95, -0.51, -0.44, -1.25, 0.78, -1.61, -0.21, -0.9 , 0.39, -0.51, -1.18, -0.03, 0.43, 0.07, 0.3 , -0.63, -0.36, -0.67, -0.36, -0.81, -1.73, 0.18, -0.4 , -1.63, 0.46, -0.91, 0.05, 0.73, 0.13, 1.14, -1.23, 0.4 , -0.68, -0.87, -0.58, -0.31, 0.06, -1.17, 0.9 , 0.47, -1.54, 1.49, 1.9 , 1.18, -0.18, -1.07, 1.05, -0.4 , 1.22, 0.21, 0.98, 0.36, 0.71, 0.01, 1.79, 0.13, 0.4 , 1.88, -1.35, -1.27, 0.97, -1.17, 1.94, -0.41, -0.75, 1.92, 1.48, 1.87, 0.91, -0.86, 1.91, -0.27, 0.8 , 0.95, -0.16, 0.61, 0.92, 0.38, -1.1 , 0.3 , 1.33, -0.69, -0.15, -0.44, 1.85, 0.67, 0.41, -0.77, 0.54, -0.67, 0.03, -0.64, 0.68, 0.58, -0.21, 0.4 , -1.09, -1.49, 0.44, 0.17, 0.64, 2.38, 0.94, -0.91, 1.12, -1.32, -0.46, -0.07, 1.71, -0.74, -0.83, -0.1 , -0.66, 1.13, -1.08, -1.15, -0.44, -0.5 , 1.93, 0.95, 0.09, -1.23, 0.84, -1. , -1.54, 1.19, 0.32, 0.92, 0.32, 0.86, -0.65, -1.03, 0.68, -0.8 , -0.69, -0.46, 0.02, -0.35, -1.37, -0.64, -2.22, 0.63, -1.6 , -1.1 , 0.05, -0.74, 1.54, -1.29, 0.27, -0.04, -1.17, 0.52, -0.17, 0.77, 0.82, 2.16, 1.34, -0.37, -0.24, 1.1 , 0.66, 0.64, -1.62, -0.02, -0.74, 0.28, -0.1 , 0.91, 0.32, 0.79, -0.47, -0.94, -0.41, -0.02, 0.38, 2.26, -0.04, -0.96, -0.35, -0.46, 0.48, -1.54, 0.06, 0.16, 0.23, -0.6 , -0.24, -1.42, -0.49, -0.54, 0.42, -1.16, 0.78, 1.49, -2.07, 0.43, 0.68, -0.64, -0.4 , -0.13, -0.3 , -0.31, -1.68, 1.15, 1.08, -0.81, -1.47, 0.52, -0.58, 0.14, -0.32, 0.69, 0.69, -0.73, -1.38, -1.58, 0.61, -1.19, -0.51, -0.6 , -0.05, -1.94, 0.19, 0.52, 0.09, -0.31, 0.1 , 0.4 , -2.77, 1.96, 0.39, -0.65, -0.39, 0.49, -0.12, -2.03, 2.06, -0.11], [ 1.02, -0.69, 1.54, 0.29, 0.61, -1.05, 1.21, 0.69, 1.3 , -0.63, -0.48, 2.3 , -1.06, -0.14, 1.14, 0.1 , 0.58, -0.4 , 0.37, -1.31, 1.66, -0.12, -0.68, 0.67, -0.46, -1.33, -1.35, 0.69, -0.16, -0.13, 1.08, -1.13, -0.73, -0.38, 0.09, -0.04, -0.29, -0.06, -0.11, -0.72, -0.81, 0.27, -0.89, -1.16, -0.31, -0.16, 2.26, -0.7 , 0.94, 0.75, -1.19, 0.77, -1.18, -2.66, 0.61, -1.76, 0.45, -0.68, 1.66, 1.07, -0.45, -0.69, -1.21, -0.44, -0.28, -0.36, 0.16, 0.58, 0.35, -0.76, -1.44, 1.36, -0.69, -0.65, -0.52, -1.84, -0.48, -0.48, 0.62, 0.7 , 0. , 0.93, 0.34, -0.02, 0.16, -0.19, -0.39, -0.27, -1.13, 0.28, -0.99, 0.84, -0.25, 0.05, 0.49, 0.64, -1.57, -0.21, 0.88, -1.7 , 0.39, -2.26, -1.02, 0.04, -1.66, -0.99, -1.47, 1.65, 0.16, 0.57, -0.22, -0.35, -1.62, -0.29, -0.76, 0.86, 1.14, 1.47, 0.85, -0.6 , -1.12, 0.77, 0.36, -1.77, 0.36, 0.81, 0.06, -0.19, -0.81, -1.45, 0.8 , -0.31, -0.23, 1.73, 0.68, 0.37, 0.14, 1.52, 1.72, 0.93, 0.58, -2.09, 0.12, -0.13, 0.09, 0.94, -2.74, -0.57, 0.27, -0.47, -1.42, 0.87, 0.28, -0.97, 0.31, 0.82, 0.01, 0.8 , 0.08, -0.4 , -1.16, -0.09, 0.19, 0.88, -0.12, 0.46, -0.96, -0.78, -0.11, -1.05, 0.82, 0.46, 0.28, 0.34, 2.02, -0.47, -2.2 , 0.2 , -0.05, -0.52, -0.98, -0.44, 0.18, -0.5 , 2.41, -0.96, -0.79, -2.29, 0.25, -2.02, -0.54, -0.28, -0.71, 1.74, 0.99, 1.32, -0.88, 1.13, 0.5 , 0.77, 1.03, -0.91, -0.42, 0.86, -2.66, 1.51, 0.55, -0.05, 0.22, -1.03, -0.35, 1.1 , 1.3 , 2.7 , -0.07, -0.66, -0.51, -1.02, -0.08, 0.38, -0.03, 1.1 , -0.23, -0.35, -0.58, -1.63, -1.57, -1.18, 1.3 , 0.9 , 1.37, -1.33, -1.97, -0.66, 0.18, 0.5 , 1.05, 0.28, 1.74, -0.22, -0.91, -1.68, -0.89, 0.24, -0.89, 0.94, 1.41, -2.37, 0.86, -2.24, 0.4 , 1.22, 0.06, -1.28, -0.59, -0.26, -0.18, -0.2 , -0.11, 0.21, -1.21, -0.24, 1.52, -0.38, -0.44, 1.08, -2.56, 1.18, -0.63, 0.16, 0.1 , 0.94, -0.27, -0.68, 1.3 , -2.36, 0.02, -1.35, -0.76, 2.01, -0.04], [ 0.2 , -1.78, -0.73, 0.2 , 0.35, 0.62, 0.01, 0.53, 0.45, -1.83, 0.04, 0.77, 0.59, -0.36, -0.81, -1.12, -0.13, 1.13, -1.95, -0.66, -1.14, 0.78, -0.55, -0.47, -0.22, 0.45, -0.39, -3.05, 0.54, 0.44, -0.22, -1.08, 0.35, 0.38, -0.47, -0.22, -0.93, -0.18, -1.55, 0.42, -0.94, 0.24, -1.41, -0.59, -0.11, -1.66, 0.12, -0.38, -1.74, -1.3 , 0.61, 0.9 , -0.13, 0.4 , 0.22, 0.33, 1.29, -1.51, 0.68, -0.38, -0.22, -0.3 , -0.38, -1.23, 0.18, 1.67, -0.06, -0. , -0.69, -0.12, 0.47, -0.37, -0.45, 0.4 , -0.92, 0.25, 0.82, 1.36, -0.09, 1.37, 1.03, -1. , -1.22, -0.3 , 1.03, -0.07, -0.6 , 1.55, 0.29, -2.32, 0.32, 0.52, 0.23, 0.45, -0.07, -1.32, -0.37, -0.95, -0.93, -1.26, 0.45, 0.1 , -0.45, -0.65, -0.02, 1.08, -2. , 0.38, -0.55, -1.88, -1.95, -0.91, 0.22, 0.39, -0.94, 1.02, 1.42, 0.4 , -0.59, 1.12, 0.76, 0.87, -0.66, -2.83, 2.12, -1.61, -0.04, 2.38, 0.33, 0.95, -1.5 , -1.78, -0.53, 1.09, -0.35, -0.79, 0.2 , 1.08, -1.44, -1.21, -0.79, 1.09, 0.23, 2.13, 0.94, -0.04, 1.27, 0.21, -0.7 , 0.68, -0.7 , -0.29, 1.33, -0.1 , -0.8 , -0.46, 1.02, -0.55, -0.39, -0.51, 0.18, -0.39, -1.6 , -0.89, -0.93, 1.24, 0.81, 0.59, -0.51, -0.82, -0.51, -1.05, 2.5 , -2.25, 0.56, -1.28, -0.1 , -0.99, -1.18, -1.14, 1.75, -0.13, -0.77, 0.56, 0.01, 0.72, -1.82, 0.3 , 0.77, -1.66, 0.45, 1.7 , -0.01, 0.82, 0.67, -0.71, 0.04, -1.57, -0.45, 0.27, 0.72, 0.02, 0.72, -1.1 , -0.1 , 0.02, 1.85, -0.21, -0.5 , 0.02, -0.92, 0.19, -0.37, -1.79, -0.06, -0.32, -1.63, -0.07, 1.49, 0.52, 0.61, -1.34, 0.48, 0.15, 0.53, 0.42, -1.36, -0.04, -0.76, -0.05, -0.9 , 1.31, -0.86, -0.9 , 0.07, -1.08, -0.42, -0.83, 1.41, 0.79, -0.06, -0.39, 0.94, 0.41, 0.5 , -0.03, -1.69, -0.11, -0.53, 0.65, 1.01, -0.66, 0.47, 1.74, -0.67, 1.68, -0.85, 0.02, -0.01, 0.01, -0.84, -0.59, -0.67, 0.33, 0.33, 2.23, 1.37, -0.51, 0.32, 1. , 0.03, -0.07, 0.05, 0.87, -0.85, -0.33, 0.47, 0.31, 0.24, -0.37, 0.97], [ 2.13, 0.41, -0.19, 0.76, -0.54, -0.75, 0.03, -2.58, -1.15, -0.35, -1.35, -1.03, -0.44, -1.64, -0.41, -0.54, 0.03, 1.15, 0.17, 0.02, 0.1 , 0.23, -1.02, -0.11, 0.31, -1.37, 0.87, 1.08, -0.63, -0.24, -0.88, 0.7 , -1.06, -0.22, -0.86, 0.05, -1.79, 1.33, -0.96, 0.06, -0.21, -0.76, -0.89, 0.94, -0.53, 0.27, -0.8 , -0.65, 0.47, 0.93, -0.18, -1.42, 2. , -0.86, -1.54, 2.59, -0.4 , -1.46, -0.68, 0.37, 0.19, -0.85, 1.82, -0.52, -1.18, 0.96, 1.33, -0.82, -1.4 , 1.03, -2.05, -1.23, 0.97, -0.06, -0.26, 0.35, -0.15, -1.3 , 1.28, 1.33, 0.21, 0.05, 2.34, -0.28, -0.26, 0.36, 1.47, 1.59, -0.26, 0.31, -1.38, -0.31, -0.84, -1.01, 1.68, -0.79, -0.53, 0.37, 1.3 , 0.48, 2.76, -0.07, 0.26, 0.28, 1.44, 0.51, -0.12, -0.95, 0.24, 1.4 , -0.41, 0.53, 0.25, 0.86, -0.8 , 2.35, -1.28, -0.37, 0.94, 0.3 , 0.83, -0.5 , -0.07, 0.01, 1.57, 0.69, 0.8 , -0.66, 0.97, 0.23, 1.39, 2.01, -0.31, -0.41, -0.86, -0.14, -0.38, 0.36, -0.14, -0.36, 1.06, -0.94, 0.43, -0.41, 0.72, 1.39, -0.3 , 0.44, 0.18, -0.8 , 0.24, 0.29, 0.41, -0.2 , 0.09, -1.15, -0.36, 0.56, 0.89, -0.42, 0.1 , 0.23, 0.2 , 0.54, -1.82, -0.05, 0.24, -1. , 1.67, 0.16, 1.56, -0.79, -0.91, 0.22, -1.68, 0.21, 0.1 , 1.02, 0.7 , -0.42, -1.1 , 1.71, -0.79, -1.05, -1.08, 1.12, -0.52, -0.75, 0.14, -0.21, -0.68, 0.75, 1.07, 0.99, 0.77, 0.4 , -1.78, 1.67, 0.3 , 0.61, 1.11, 1.43, 0.42, 0.44, -0.6 , 0.03, -0.85, -0.72, -0.89, -0.16, 1.05, 3.17, 0.19, -1.35, 1.26, -0.3 , -0.66, 0.21, -1.24, 0.22, -0.09, 0.1 , 0.38, 0.07, 0.02, 0.28, 0.42, -1.03, -1.43, -0.06, -1.43, 0.09, 0.94, 0.61, -1.05, -0.86, 0.33, -0.4 , -0.32, 0.6 , -0.99, -0.4 , -0.8 , -1.04, -0.86, 0.68, 0.05, -0.88, -0.23, -1.64, -0.73, 2.15, -0.09, 0.73, -0.07, 0.35, 0.66, -1.1 , -0.03, 1.58, -0.8 , -0.57, -0.31, 0.27, 0.52, 1.27, 0.5 , -0.06, 1.26, 0.7 , -1.5 , 2.53, 1.77, -0.17, 0.38, 1.32, -0.17, 0.73, 1.1 , -1.01, -0.6 ], [ 0.92, 0.46, 0.92, -0.13, -0.29, -2. , -1.15, 0.05, 0.82, 0.53, -0.13, -0.27, 0.22, 0.08, 1.4 , 0.15, -1.48, -1.27, 1.52, -1.17, 0.76, -0.27, -0.17, -0.13, 1.22, -0.19, -0.03, -1.53, 0.21, 0.53, 0.24, 1.4 , 0.06, 0.3 , 1.65, -1.55, -0.46, 1.43, 0.94, 0.68, 0.83, 0.33, 1.63, 0.38, 0.24, 0.16, 0.19, -1.16, 0.77, -0.13, 1.82, -0.08, 0.42, 0.25, -0.63, 0.99, 1.91, -0.01, -0.3 , -0.36, -1.89, -0.18, 0.25, 1.05, 0.96, -0.42, -0.28, 1.12, -0.17, -0.51, 1.39, 1.04, 0.02, -0.59, -2.01, 0.59, -0.9 , -1.96, 1.58, 0.65, -1.14, -1.21, 0.87, -0.88, 1.3 , 0.62, 0.54, 0.4 , 0.19, 0.88, -0.45, 0.09, 0.75, 0.56, -1.19, -0.5 , 0.25, -0.41, 1.77, -0.39, -0.16, 0.77, 0.33, -0.15, -0.76, 0.3 , 1.04, 0.48, -0.78, 1.74, -1.45, -1.58, 0.96, 0.23, -0.55, -1.1 , 2.32, 0.12, 0.53, 0.32, 0.43, 0.54, 0.73, -0.38, -0.29, -1.74, -0.78, 0.27, 1.05, 0.6 , -0.34, -1.26, -2.78, 1.15, -0.59, -0.45, 0.13, -1.41, -0.35, 2.02, 0.51, 0.36, -1.58, 2.24, -1.42, 1.92, -2.12, 1.41, 1.62, -0.82, 0.42, 0.55, -0.81, -1.45, -1.32, 0.54, -0.09, -0.56, 0.97, 0.51, -0.76, -1.2 , 0.52, -0.54, 0.1 , 1.58, 0.5 , -0.86, 0.16, -0.95, 1.61, -0.56, 0.21, 0.31, 0.16, -1.96, -1.45, -0.45, 0.32, -0.14, -0.96, -1.35, -0.4 , -0.47, 0.51, -0.33, 0.6 , -0.59, -0.26, -0.35, -0.78, 0.63, -0.81, -0.52, -0.07, -1.3 , -0.32, -0.71, -0.39, -0.06, -0.8 , -0.22, 1.31, -0.03, 1.15, 0.35, 0.77, -0.77, 0.1 , 0.13, -0.61, -0.82, -1.49, 1.5 , -0.97, 1.35, -0.47, -0.86, 0.62, -0.63, 0.57, -0.33, 0.48, -0.97, 0.83, 0.49, -0.92, 2.64, 0.54, 2.29, 1.6 , -0.19, -0.41, -0.4 , -1.83, -0.7 , 0.25, 1.53, -0.77, 0.88, -1.25, -0.59, -0.46, 0.37, 0.46, 0.96, 0.77, 0.24, 0.39, 1.59, -0.51, 0.77, -1.81, 0.41, -0.48, 0. , 1.04, 0.16, 0.89, 1.47, 0.39, 1.17, -0.33, -0.01, -0.52, 1.04, 0.41, -0.51, 0.15, 1.04, -0.04, -0.95, 0.13, -1.98, 0.77, -0.42, -0.47, 0.88, -1.37, 1.95, -0.48]]), 'b1': np.array([[-0.52], [ 1.02], [ 0.71], [ 2.45], [-0.21]]), 'W2': np.array([[-0.12, -1.48, -0.33, -0.72, -0.45], [-1.74, 1.66, -1.42, -2.8 , -1.19], [-0.6 , -1.15, 1.1 , -0.14, 0.02], [ 0.61, 0.28, 0.98, -1.11, -0.55], [ 0.67, -2.54, -1.38, 0.5 , -0.48]]), 'b2': np.array([[ 0.94], [ 0.81], [-1.2 ], [ 0.41], [ 1.2 ]]), 'W3': np.array([[ 0.15, -0.98, 0.88, 0.63, 0.54], [ 0.72, -2.99, 0.88, 1.81, 0.44], [ 0.19, 0.7 , 0.34, 0.65, -0. ], [-0.77, -1. , -1. , -1.37, -1.07], [ 1.76, 0.75, -0.62, -0.39, 0.11]]), 'b3': np.array([[-0.67], [ 0.07], [ 0.78], [-0.03], [ 0.34]]), 'W4': np.array([[ 0.86, -0.31, 0.24, -0.33, -0.07]]), 'b4': np.array([[-0.38]])}
    activations = forward_propagation(X,parametres) 
    nbCouche = len(parametres) // 2 
    Af = activations['A'+str(nbCouche)] 
    return Af >= 0.5

def create_all_variables():
    delete_flask_files()


    path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

    files = os.listdir(path_to_project+"/1. Data/temp PDF")
    poppler_path = path_to_project+"/poppler-21.09.0/bin"

    start_time = time.monotonic()
    print("Création des images (et images pixelisées)...")

    create_all_images()
    pixelize_images()

    end_time = time.monotonic()
    print("Création des images (et images pixelisées): {:.2f} seconds".format(round(end_time - start_time, 2)))

    start_time = time.monotonic()
    print("Création des variables...")

    pixelated_image_names = os.listdir(path_to_project+"/1. Data/temp pixelated images/")

    datas = {}

    for file_name in pixelated_image_names:
        path = path_to_project+"/1. Data/temp pixelated images/"+file_name
        data = get_colors(path)
        datas[data[0]] = data[1:]

    all_image = os.listdir(path_to_project+'/1. Data/temp images')
    dico = create_characters_values(all_image)

    # Columns in the right order
    for image in dico:
        if(image in datas):
            number_colors = datas[image][0]
            datas[image].pop(0)
            dico[image].insert(0,number_colors)
            dico[image].extend(datas[image])

    end_time = time.monotonic()
    print("Création des variables: {:.2f} seconds".format(round(end_time - start_time, 2)))

    print("\n\n")

    print(dico)

    for imagei in all_image:  
        print(imagei,predict(dico[imagei]))
        path_image = path_to_project+"/1. Data/temp images/" + str(imagei)
        if dico[imagei][0]>0.2:
            shutil.move(path_image, path_to_project+"/Flask App/static/image/"+ str(imagei))
        else :
            shutil.move(path_image, path_to_project+"/Flask App/static/texte/"+ str(imagei))

    delete_temp_files()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "supersecretkey"

# ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save("/home/bbordenave/personalProjects/PredictionImagesInPDF/1. Data/temp PDF/"+filename)
            flash('File successfully uploaded')
            flash('File push')
            create_all_variables()
            flash('Variables creation...')
            return redirect(url_for('display_files'))
        else :
            print("FILES NOT ALLOWED")
    return render_template('upload.html')

@app.route('/display')
def display_files():
    image_files = []
    text_files = []

    image_path = os.path.join("static", "image")
    text_path = os.path.join("static", "texte")

    for file in os.listdir(image_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            image_files.append(file)

    for file in os.listdir(text_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            text_files.append(file)

    return render_template('display_files.html', image_files=image_files, text_files=text_files)

if __name__ == '__main__':
    app.run(debug=True)
