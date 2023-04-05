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

path_to_project = "/home/bbordenave/personalProjects/PredictionImagesInPDF"

files = os.listdir(path_to_project+"/1. Data/temp PDF")
poppler_path = path_to_project+"/poppler-21.09.0/bin"

def delete_temp_files():
    files = glob.glob(path_to_project+'/1. Data/temp images/*')
    for f in files:
        os.remove(f)

    files = glob.glob(path_to_project+'/1. Data/temp pixelated images/*')
    for f in files:
        os.remove(f)

    files = glob.glob(path_to_project+'/1. Data/temp PDF/*')
    for f in files:
        os.remove(f)

def create_images(image, file):
    """Create a jpeg in the directory './Labelisation'"""
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
    image_names = os.listdir(path_to_project+"/1. Data/temp images/")
    for image in image_names:
        path = path_to_project+'/1. Data/temp images/'+image
        create_pixel_image(path,path_to_project+"/1. Data/temp pixelated images/")

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

def get_colors(imagePath):
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
    dico_images = {}
    for imagei in files_of_image:  
        path = path_to_project+"/1. Data/temp images/" + str(imagei)
        img = cv2.imread(path, 0)
        
        result_txt = pytesseract.image_to_string(img)
        a = len(result_txt)
        dico_images[imagei] = [a] # nb chars    
        
        x,y = img.shape        
        shape = x*y        
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
        dico_images[imagei].append(letter_somme)

        dico_images[imagei].append(round(letter_somme/shape,2))

    return dico_images


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

delete_temp_files()
