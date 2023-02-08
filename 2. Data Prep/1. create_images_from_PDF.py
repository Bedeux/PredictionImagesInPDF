import os
from pdf2image import convert_from_bytes, convert_from_path

files = os.listdir("C:/Users/ELE60800ad9b1d1a/Documents/PythonProject/PredictionImagesInPDF/1. Data/Original Files/New files")
poppler_path = "C:/Users/ELE60800ad9b1d1a/Documents/PythonProject/PredictionImagesInPDF/poppler-0.68.0/bin"

def create_images(image, file):
    """Create a jpeg in the directory './Labelisation'"""
    filename = file.replace('.pdf', '')
    for i in range(len(image)):
        if len(image) > 1:
            new_filename = 'C:/Users/ELE60800ad9b1d1a/Documents/PythonProject/PredictionImagesInPDF/1. Data/Labelisation/' + filename + '-' + str(i) + '.jpeg'
        else:
            new_filename = 'C:/Users/ELE60800ad9b1d1a/Documents/PythonProject/PredictionImagesInPDF/1. Data/Labelisation/' + filename + '.jpeg'
        image[i].save(new_filename, 'JPEG')


def create_all_images():
    """Creates images in the directory images from the PDFs"""
    n = 0
    print(n)
    for file in files:
        image = convert_from_path("C:/Users/ELE60800ad9b1d1a/Documents/PythonProject/PredictionImagesInPDF/1. Data/Original Files/New files/" + file, 500, poppler_path=poppler_path, use_pdftocairo=True, strict=False)
        create_images(image, file)
        n+=1
        print(n)

create_all_images() # From PDF, creates images of all the pages