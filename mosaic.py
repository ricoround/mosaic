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
        target_path,
        tile_directory,
        output_image,
        tile_size=32,
        grid_size=4,
        reuse_images=True,
    ):
        self.tile_directory = tile_directory
        self.output_image = output_image
        self.tile_size = tile_size
        self.grid_size = grid_size
        self.reuse_images = reuse_images
        self.tiles = []

        

        # Load the target image
        target_image = cv2.imread(target_path)
        height, width = target_image.shape[0], target_image.shape[1]
        min_size = min(height, width)
        tile_size = min_size // self.grid_size
        print(f"tile_size: {tile_size}")

        # How many tiles fit in the image. Always an integer.
        self.grid = [height // tile_size, width // tile_size] 

        # Create the grid
        self.grid_height =  self.grid[0] * tile_size
        self.grid_width = self.grid[1] * tile_size
        print(f"grid: {self.grid}, wh:{self.grid_width, self.grid_height}")
        self.target_image = target_image[0: self.grid_height,0: self.grid_width]
        print(f"target_image: {self.target_image.shape}")

        # Tile size
        self.tile_size = tile_size
        self.image = self.target_image.copy()

        print(f"target_image: {self.target_image.shape}")
        print(f"image: {self.image.shape}")

    def generate(self):
        self.load_tiles(self.tile_directory)
        self.print_info()
        self.fit_tiles()

        # self.load_samples()
        # self.create_tiles()
        # self.resize_image()
        # self.create_grid()
        # self.fill_grid()
        # self.save_image()
        return self.image
    
    def print_info(self):
        print_msg(f"Tile directory: {self.tile_directory}", INFO)
        print_msg(f"Target image: {self.target_image.shape}", INFO)
        print_msg(f"Grid size: {self.grid_size}", INFO)
        print_msg(f"Tile size: {self.tile_size}", INFO)
        print_msg(f"Number of tiles: {len(self.tiles)}", INFO)
        print_msg(f"Grid: {self.grid}, wh:{self.grid_width, self.grid_height}", INFO)
        print_msg(f"Grid tiles: {self.grid[0] * self.grid[1]}", INFO)

        return

    def load_tile(self, path):
        # Check if the file is an supported image format
        if path.endswith(ACCEPTED_IMAGE_FORMATS):
            # Load the image using cv2
            img = cv2.imread(path)
            if img is not None:
                self.tiles.append(Tile(path, img, self.tile_size))
                print_msg(f"Loaded tile: {path}", SUC)

        return

    def load_tiles(self, tile_dir):
        # Loop through the files and folders in the folder
        for root, subfolder, filename in os.walk(tile_dir):
            # Check if there is a folder inside the folder
            # for dir in subfolder:
            #     # If there is a folder inside the folder, take the images from that folder
            #     self.load_tiles(os.path.join(root, dir))

            for file in filename:
                self.load_tile(os.path.join(root, file))

            print_msg(f"Loaded {len(self.tiles)} tiles", INFO)

        return
    
    def get_best_tile(self, i, j):
        # Get the best tile for the position
        # Calculate the average color of the target tile
        target_tile = self.target_image[i * self.tile_size : (i + 1) * self.tile_size, j * self.tile_size : (j + 1) * self.tile_size]
        print(f"target_tile: {target_tile.shape}")
        target_average_color = np.mean(target_tile)

        tiles_averages = [tile.average_color for tile in self.tiles]

        differences = np.abs(tiles_averages - target_average_color)
    
        # Find the index of the minimum absolute difference
        best_tile_index = np.argmin(differences)
        
        # Append the closest value and its index to the respective lists
        # closest_values.append(second_array[min_index])
        # closest_indices.append(min_index)


        # Return the best tile
        return self.tiles[best_tile_index]


    def add_tile(self, tile, y, x):
        # Add the tile to the image
        # Calculate the position of the tile
        # x = i * self.tile_size
        # y = j * self.tile_size

        print(f"tile: {tile.tile.shape}")
        print(f"image: {self.image.shape}")

        print(f"x: {x}, y: {y}")

        test = self.image[y * self.tile_size : (y+1) * self.tile_size, x * self.tile_size : (x+1) * self.tile_size]
        print(f"test: {test.shape}")


        # Add the tile to the image
        self.image[y * self.tile_size : (y+1) * self.tile_size, x * self.tile_size : (x+1) * self.tile_size] = tile.tile

        return

    def fit_tiles(self):
        # Loop through the grid and find the best tile for each position
        # Loop through the rows
        for j in range(self.grid[0]):
            # Loop through the columns
            for i in range(self.grid[1]):
                # Get the tile that fits best
                print(f"i: {i}, j: {j}")
                tile = self.get_best_tile(j, i)

                # Add the tile to the image
                self.add_tile(tile, j, i)


class Tile:
    def __init__(self, path, image, tile_size=32):
        self.path = path
        self.image = image
        self.tile_size = tile_size
        self.create_tile()
        self.average_color = self.get_average_color()

    def __repr__(self):
        return f"Tile({self.path})"

    def create_tile(self):
        # Create a tile from the image
        # Resize the image to the square tile size
        size = self.image.shape[0], self.image.shape[1]
        min_size = min(size)
        # Create a square image from the left top corner of the image
        img = self.image[:min_size, :min_size]
        self.tile = cv2.resize(img, (self.tile_size, self.tile_size))
        return

    def get_average_color(self):
        # Calculate the average color of the tile
        return np.mean(self.tile)





# Define a function to compare images and find the best one
def compare_images(target_values, overlay_image):
    # Compute a metric to determine how well the overlay image matches the target values
    # Calculate the absolute difference between the overlay image and target values
    diff = np.abs(overlay_image - target_values).mean()
    return diff



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
    return img[
        middle_x - half_size : middle_x + half_size,
        middle_y - half_size : middle_y + half_size,
    ]


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
    for filename in os.listdir(folder_path):
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

    if len(sys.argv) < 4:
        print_msg(
            "You must provide an image and a folder of images.\nExample: python photomosaic.py img.jpg my_photos 16",
            ERR,
        )
        sys.exit(1)

    input_img_filename = sys.argv[1]
    tile_imgs_foldername = sys.argv[2]
    grid_size = int(sys.argv[3])

    # Check if the image exists
    if not os.path.isfile(input_img_filename):
        print_msg(f"Error: {input_img_filename} is not a valid file", ERR)
        sys.exit(1)

    # Check if the folder exists
    if not os.path.isdir(tile_imgs_foldername):
        print_msg(f"Error: {tile_imgs_foldername} is not a valid directory", ERR)
        sys.exit(1)

    mosaic = PhotoMosaic(
        input_img_filename,
        tile_imgs_foldername,
        "output.jpg",
        grid_size=grid_size,
    )

    mosaic.generate()
    cv2.imwrite("test_result.jpg", mosaic.image)
