import os
import csv

with open('Merge CSVs/targetValues.csv', 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    all_images = os.listdir("./Labelisation/Images")
    for image in all_images:
        csv_writer.writerow([image, 1])

    all_images = os.listdir("./Labelisation/Textes")
    for image in all_images:
        csv_writer.writerow([image, 0])
