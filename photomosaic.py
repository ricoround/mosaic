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

ACCEPTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")


class PhotoMosaic:
    def __init__(
        self,
        target_image,
        tile_directory,
        output_image,
        tile_size=32,
        grid_size=8,
        reuse_images=True,
    ):
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


class Tile:
    def __init__(self, path, image):
        self.path = path
        self.image = image
        self.average_color = self.get_average_color(image)

    def __repr__(self):
        return f"Tile({self.path})"

    def get_average_color(self):
        # Calculate the average color of the image
        average_color = np.average(self.image, axis=(0, 1))
        return average_color


class TileManager:
    def __init__(self, tile_directory, tile_size, reuse_images):
        self.tile_directory = tile_directory
        self.tile_size = tile_size
        self.reuse_images = reuse_images
        self.tiles = []


def crop_to_square(imgs):
    # Crop images to squares
    tiles = []
    for img in imgs:
        # Crop the image to a square from the middle of the image
        # Take the minimum of the width and height and use that as the size of the square
        size = min(img.shape[0], img.shape[1])

        # Calculate the top left corner of the crop
        x = (img.shape[1] // 2) - (size // 2)
        y = (img.shape[0] // 2) - (size // 2)

        img = img[y : y + size, x : x + size]
        tiles.append(img)
    return tiles

# Define a function to compare images and find the best one
def compare_images(target_values, overlay_image):
    # Compute a metric to determine how well the overlay image matches the target values
    # Calculate the absolute difference between the overlay image and target values
    diff = np.abs(overlay_image - target_values).mean()
    return diff


def split_img_to_tiles(img, tile_size):

    # Split the image into tiles
    tiles = []
    for i in range(0, img.shape[0], tile_size):
        for j in range(0, img.shape[1], tile_size):

            tile = img[i : i + tile_size, j: j + tile_size]
            tiles.append(tile)

    return tiles



def fit_tiles_to_img(input_tiles, output_tiles):

    output_img = input_tiles.copy()

    for tile in input_tiles:
        print(tile.shape)
    
    # return output_img

    # Loop through the tiles and find the best one for each tile
    for i, tile in enumerate(input_tiles):
        best_overlay = None
        best_diff = float("inf")

        # Iterate through overlay images to find the best one
        for overlay_image in tqdm(output_tiles):
            diff = compare_images(tile, overlay_image)
            # print(overlay_image)
            if diff < best_diff:
                best_diff = diff
                best_overlay = overlay_image

        # Overlay the best image onto the background
        print(best_overlay.shape)
        print(tile.shape)
        output_img[i] = cv2.add(tile, best_overlay)
        # output_img[i] = best_overlay

    return output_img


def square_crop(img):
    # Crop the image to a square from the middle of the image
    # Take the minimum of the width and height and use that as the size of the square. Make sure the size is even.
    width, height, _ = img.shape
    size = min(img.shape[0], img.shape[1])
    print(size)
    size = size - (size % 2)
    print(size)
    middle_x = width // 2
    middle_y = height // 2
    half_size = size // 2
    print_msg(f"centerx = {middle_x}, centery = {middle_y}, halfsize={half_size}", INFO)
    return img[middle_x - half_size : middle_x + half_size, middle_y - half_size : middle_y + half_size]


def tiles_to_img(fitted_tiles, split):
    result = []
    for i in range(0, len(fitted_tiles), split):
        print(i)
        result.append(np.hstack(fitted_tiles[i : i + split]))
    
    return np.vstack(result)


def mosaic(input_img_filename, tile_imgs_foldername):

    input_img = load_img(input_img_filename)
    tile_imgs = load_imgs(tile_imgs_foldername)

    # Crop images to squares
    input_img = square_crop(input_img)
    
    print(input_img.shape)
    output_tiles = crop_to_square(tile_imgs)

    # Split the image into tiles
    split = 2**3
    tile_size = input_img.shape[0] // split
    print(input_img.shape[0])
    print_msg(f"Tilesize: {tile_size}", INFO)

    # Make output tiles as large as the input tiles
    output_tiles = [cv2.resize(tile, (tile_size, tile_size)) for tile in output_tiles]


    input_tiles = split_img_to_tiles(input_img, tile_size)

    fitted_tiles = fit_tiles_to_img(input_tiles, output_tiles)
    
    result = tiles_to_img(fitted_tiles, split)

    cv2.imwrite("test.jpg", result)

    return



def load_img(img_path):
    try:
        # Load the image using cv2
        image = cv2.imread(img_path)

        print_msg(f"Loaded image {img_path}", SUC, True)
        return image

    except Exception as e:
        print_msg(f"Error loading image {img_path}\n{e}", ERR)
        return None


def load_imgs(folder_path):
    images = []


    # Loop through the files and folders in the folder
    for filename in tqdm(os.listdir(folder_path)):

        # Check if there is a folder inside the folder
        if os.path.isdir(os.path.join(folder_path, filename)):
            # If there is a folder inside the folder, take the images from that folder
            images += load_imgs(os.path.join(folder_path, filename))
        

        # Check if the file is an image
        if filename.endswith(ACCEPTED_IMAGE_FORMATS):
            # Construct the full path to the image
            img_path = os.path.join(folder_path, filename)

            # Load the image using cv2
            img = load_img(img_path)
            if img is not None:
                images.append(img)

    return images


def print_msg(msg, msg_type=None, tqdm_bar=None):
    if msg_type == INFO:
        msg_col = colored(msg, "blue")
    elif msg_type == SUC:
        msg_col = colored(msg, "green")
    elif msg_type == WARN:
        msg_col = colored(msg, "yellow")
    elif msg_type == ERR:
        msg_col = colored(msg, "red")
    else:
        msg_col = msg

    if tqdm_bar:
        tqdm.write(msg_col)
    else:
        print(msg_col)
    return


if __name__ == "__main__":
    # Create mosaic image from a given image and a folder of images.
    # Usage: python photomosaic.py <image> <folder>
    # Example: python photomosaic.py mona_lisa.jpg my_photos

    print_msg("Photo Mosaic Generator", INFO)
    print_msg("----------------------", INFO)

    if len(sys.argv) < 3:
        print_msg(
            "You must provide an image and a folder of images.\nExample: python photomosaic.py img.jpg my_photos",
            ERR,
        )
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
