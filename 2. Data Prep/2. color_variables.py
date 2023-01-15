from PIL import Image
from collections import Counter
import numpy as np
import scipy.spatial as sp
import cv2
import csv
import os
from skimage import io

def create_pixel_image(imagePath, newDirectory):
    img = Image.open(imagePath)
    A4_proportion = 297 / 210
    n = 25 # Number of pixel in the larger of the page
    imgSmall = img.resize((n, int(n * A4_proportion)), resample=Image.Resampling.BILINEAR)
    imageName = imagePath.split('/')[-1]
    newPath = newDirectory + imageName
    imgSmall.save(newPath)

def pixelize_images():
    image_names = os.listdir("./Labelisation/Images")
    for image in image_names:
        path = './Labelisation/Images/'+image
        create_pixel_image(path,"./Labelisation/Images Pixelisées/")

def pixelize_textes():
    image_names = os.listdir("./Labelisation/Textes")
    for image in image_names:
        path = './Labelisation/Textes/'+image
        create_pixel_image(path,"./Labelisation/Textes Pixelisées/")

# pixelize_images()
# pixelize_textes()

# Stored all RGB values of main colors in a array
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
    return [imageName, ColorNumber, MostFrequentColor, SecondMostFrequentColor]

image_names = os.listdir("./Labelisation/Textes Pixelisées")

with open('./data_AI.csv', 'a+', newline='') as write_obj:
    for file_name in image_names:
        path = "./Labelisation/Textes Pixelisées/"+file_name
        csv_writer = csv.writer(write_obj)
        data = get_colors(path)
        csv_writer.writerow(data)
