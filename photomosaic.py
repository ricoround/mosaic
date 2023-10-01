# This script will generate photo mosaic from a given image and a folder of images.
# Using cv2 instead of PIL because opencv is faster.



import os
import sys
import random
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageChops
from PIL import ImageStat

import cv2
import numpy as np
from colorama import Fore, Back, Style
from termcolor import colored, cprint
from tqdm import tqdm

INFO = 0
WARN = 1
ERR = 2
SUC = 3

ACCEPTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')



class PhotoMosaic:
    def __init__(self, target_image, tile_directory, output_image, tile_size=32, grid_size=8, reuse_images=True):
        self.target_image = target_image
        self.tile_directory = tile_directory
        self.output_image = output_image
        self.tile_size = tile_size
        self.grid_size = grid_size
        self.reuse_images = reuse_images

        self.tiles = []
        self.image_width = 0
        self.image_height = 0
        self.grid = []
        self.grid_width = 0
        self.grid_height = 0
        self.image = None

    def generate(self):
        self.load_tiles()
        self.resize_image()
        self.create_grid()
        self.fill_grid()
        self.save_image()

    


def mosaic(input_img_filename, tile_imgs_foldername):
    # Load the background image and the array of overlay images
    # background_image = cv2.imread('image-4.jpg')
    # overlay_images = [cv2.imread('image-1.jpg'), cv2.imread('image-2.jpg'), cv2.imread('image-3.jpg')]

    input_img = load_img(input_img_filename)
    tile_imgs = load_imgs(tile_imgs_foldername)

    return



    # Define a function to compare images and find the best one
    def compare_images(target_values, overlay_image):
        # Compute a metric to determine how well the overlay image matches the target values
        # Calculate the absolute difference between the overlay image and target values
        diff = np.abs(overlay_image - target_values).mean()
        return diff

    best_overlay = None
    best_diff = float('inf')

    # Iterate through overlay images to find the best one
    for overlay_image in tqdm(overlay_images):
        diff = compare_images(background_image, overlay_image)
        # print(overlay_image)
        if diff < best_diff:
            best_diff = diff
            best_overlay = overlay_image

    # Overlay the best image onto the background
    result = cv2.add(background_image, best_overlay)

    # Save the result
    cv2.imwrite('result.jpg', result)


def load_img(img_path):

    try:
        # Load the image using cv2
        image = cv2.imread(img_path)
                
        # print_msg(f"Loaded image {img_path}", SUC)
        return image
    
    except Exception as e:
        print_msg(f"Error loading image {img_path}\n{e}", ERR)
        return None

def load_imgs(folder_path):
    images = []

    # Loop through the files in the folder
    for filename in tqdm(os.listdir(folder_path)):

        # Check if the file is an image
        if filename.endswith(ACCEPTED_IMAGE_FORMATS):
            # Construct the full path to the image
            img_path = os.path.join(folder_path, filename)
            
            # Load the image using cv2
            img = load_img(img_path)
            if img is not None:
                images.append(img)

    return images

def print_msg(msg, msg_type=None):
    if msg_type == INFO:
        print(colored(msg, 'blue'))
    elif msg_type == SUC:
        print(colored(msg, 'green'))
    elif msg_type == WARN:
        print(colored(msg, 'yellow'))
    elif msg_type == ERR:
        print(colored(msg, 'red'))
    else:
        print(msg)

if __name__ == '__main__':
    
    # Create mosaic image from a given image and a folder of images.
    # Usage: python photomosaic.py <image> <folder>
    # Example: python photomosaic.py mona_lisa.jpg my_photos

    print_msg('Photo Mosaic Generator', INFO)
    print_msg('----------------------', INFO)

    if len(sys.argv) < 3:
        print_msg('You must provide an image and a folder of images.\nExample: python photomosaic.py img.jpg my_photos', ERR)
        sys.exit(1)

    input_img_filename = sys.argv[1]
    tile_imgs_foldername = sys.argv[2]

    # Check if the image exists
    if not os.path.isfile(input_img_filename):
        print_msg(f"Error: {input_img_filename} is not a valid file", ERR)
        sys.exit(1)

    # Check if the folder exists
    if not os.path.isdir(tile_imgs_foldername):
        print_msg(f"Error: {tile_imgs_foldername} is not a valid directory", ERR)
        sys.exit(1)

    mosaic(input_img_filename, tile_imgs_foldername)
