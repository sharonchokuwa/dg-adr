import os
import zipfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import time
from tqdm import tqdm
import shutil

# Function to unzip all directories within the main directory
def unzip_all(main_dir):
    print('--------------unzipping folders')
    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isfile(item_path) and item.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_ref:
                zip_ref.extractall(main_dir)
                os.remove(item_path)

# Function to move all contents of each folder to one main directory
def move_contents(main_dir):
    print('--------------moving images to one directory')
    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                shutil.move(subitem_path, main_dir)
            os.rmdir(item_path)

# Function to convert images from tiff or tif to png
def convert_to_png(main_dir):
    print('--------------converting tiff to png')
    image_extensions = ['.tiff', '.tif']
    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith(tuple(image_extensions)):
            im = Image.open(item_path)
            png_path = os.path.splitext(item_path)[0] + '.png'
            im.save(png_path, 'PNG')
            os.remove(item_path)

# Function to resize all images
def resize_images(main_dir, img_size=224):
    print('--------------resizing images')
    start = time.time()
    i = 1
    for img_name in tqdm(os.listdir(main_dir)):
        if i % 100 == 0:
            print('Images done so far are: ', i)
        try:
            img = cv2.imread(os.path.join(main_dir, img_name))

            h, w = img.shape[:2]
            a1 = w/h
            a2 = h/w

            if(a1 > a2):

                # if width greater than height
                r_img = cv2.resize(img, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[1]/6)
                crop_img = r_img[0:img_size, margin:(margin+img_size)]

            elif(a1 < a2):

                # if height greater than width
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                margin = int(r_img.shape[0]/6)
                crop_img = r_img[margin:(margin+img_size), 0:img_size]

            elif(a1 == a2):

                # if height and width are equal
                r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):

                crop_img = r_img[0:img_size, 0:img_size]

            if(crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):
                new_img_path = os.path.join(main_dir, img_name) 
                cv2.imwrite(new_img_path, crop_img)
                i += 1
        except:
            print('Could not save image.')
    print("Time taken = ", time.time()-start)

if __name__ == "__main__":
    main_directory = r"ssl_datasets/"

    # Unzip all directories within the main directory
    unzip_all(main_directory)

    # Move all contents of each folder to one main directory
    move_contents(main_directory)

    # Convert images from tiff or tif to png
    convert_to_png(main_directory)

    # Resize all images
    resize_images(main_directory)
