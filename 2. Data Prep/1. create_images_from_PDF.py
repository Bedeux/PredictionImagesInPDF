import os
from pdf2image import convert_from_bytes, convert_from_path

files = os.listdir("C:/Users/ELE60800ad9b1d1a/OneDrive - LECNAM/Documents/IA - Image Documents/files")
poppler_path = "C:/Users/ELE60800ad9b1d1a/OneDrive - LECNAM/Documents/IA - Image Documents/poppler-0.68.0/bin"

def create_images(image, file):
    """Create a jpeg in the directory './Labelisation'"""
    filename = file.replace('.pdf', '')
    for i in range(len(image)):
        if len(image) > 1:
            new_filename = 'Labelisation/' + filename + '-' + str(i) + '.jpeg'
        else:
            new_filename = 'Labelisation/' + filename + '.jpeg'
        image[i].save(new_filename, 'JPEG')


def create_all_images():
    """Creates images in the directory images from the PDFs"""
    for file in files:
        image = convert_from_path("files/" + file, 500, poppler_path=poppler_path, use_pdftocairo=True, strict=False)
        create_images(image, file)

create_all_images() # From PDF, creates images of all the pages