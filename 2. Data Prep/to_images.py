import os
# from tkinter import image_types
from pdf2image import convert_from_bytes, convert_from_path

path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

files = os.listdir(path_to_project+"/1. Data/temp PDF")
poppler_path = path_to_project+"/poppler-21.09.0/bin"

print(files)

def create_images(image, file):
    """Create a jpeg in the directory './Labelisation'"""
    filename = file.replace('.pdf', '')
    for i in range(len(image)):
        if len(image) > 1:
            new_filename = path_to_project+'/1. Data/temp pixel test/' + filename + '-' + str(i) + '.jpeg'
        else:
            new_filename = path_to_project+'/1. Data/temp pixel test/' + filename + '.jpeg'
        image[i].save(new_filename, 'JPEG')

def create_all_images():
    """Creates images in the directory images from the PDFs"""
    n = 0
    print(n)
    for file in files:
        image = convert_from_path(path_to_project+"/1. Data/temp PDF/" + file, 500, use_pdftocairo=True, strict=False)
        create_images(image, file)
        n+=1
        print(n)

create_all_images()
#
#
#
#
#
#







from PIL import Image
from collections import Counter
import numpy as np
# import scipy.spatial as sp
# import cv2
# import csv
import os
# from skimage import io

def create_pixel_image(imagePath, newDirectory):
    img = Image.open(imagePath)
    A4_proportion = 297 / 210
    n = 35 # Number of pixel in the larger of the page
    imgSmall = img.resize((n, int(n * A4_proportion)), resample=Image.Resampling.BILINEAR)
    imageName = imagePath.split('/')[-1]
    newPath = newDirectory + imageName
    imgSmall.save(newPath)

def pixelize_images():
    image_names = os.listdir(path_to_project+"/1. Data/Labelisation/Images")
    n=0
    for image in image_names:
        n+=1
        print(n)
        path = path_to_project+'/1. Data/Labelisation/Images/'+image
        create_pixel_image(path,path_to_project+"/1. Data/Labelisation/Images Pixelisées/")

def pixelize_texts():
    image_names = os.listdir(path_to_project+"/1. Data/Labelisation/Textes")
    n=0
    for image in image_names:
        n+=1
        print(n)
        path = path_to_project+'/1. Data/Labelisation/Textes/'+image
        create_pixel_image(path,path_to_project+"/1. Data/Labelisation/Textes Pixelisées/")

# pixelize_images()
# pixelize_texts()