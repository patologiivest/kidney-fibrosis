import numpy as np
import cv2
import skimage
import glob
import os
from collections import defaultdict

input_dir = "./demo_images"

pattern = "*.png"  # Pattern to match filenames

# Create the output directory if it doesn't exist

# List all files in the folder that match the pattern
matching_files = glob.glob(os.path.join(input_dir, pattern))

# Extract the "number" from each filename
# biopsy_numbers = []
# for file_path in matching_files:
#     filename = os.path.basename(file_path)
#     parts = filename.split("_")
#     if len(parts) >= 2:
#         number = parts[1]  # Assuming "number" is the second part in the filename
#         biopsy_numbers.append(number)
#
# image_list = biopsy_numbers
# image_score_2 = defaultdict(dict)
# d = defaultdict(dict)
# image_score = {}

# for i in range(len(image_list)):
#     biopsy = image_list[i]
#     image_score_2 = {}
#
#     for s in range(1, 150):
#         fs = glob.glob(os.path.join(input_dir, 'im_' + biopsy + '*.png'))
#         for file in fs:
#             path_ext = os.path.split(file)
#             name_ext = os.path.splitext(path_ext[1])
#             name = name_ext[0]
#
#             name_broken = name.split('_')
#             bi_no = name_broken[0]
#             ti_no = name_broken[1] + "_" + name_broken[2]
#
#             tile_input = skimage.io.imread(file)
#             tile_shape = tile_input.shape[0] * tile_input.shape[1]
#             count_foreground_pixel = np.count_nonzero(tile_input)
#             grid_HSV = cv2.cvtColor(tile_input, cv2.COLOR_RGB2HSV)
#             mask_1 = cv2.inRange(grid_HSV, (0, s, 50), (5, 255, 255))
#             mask_2 = cv2.inRange(grid_HSV, (160, s, 20), (180, 255, 255))
#             mask_red = cv2.bitwise_or(mask_1, mask_2)
#             only_red = cv2.bitwise_and(tile_input, tile_input, mask=mask_red)
#
#             count_treshold_red = np.count_nonzero(only_red)
#             if count_treshold_red != 0:
#                 tile_score = (count_treshold_red / count_foreground_pixel) * 100
#                 d[biopsy][ti_no] = tile_score
#
#         image_score = {k: sum(v.values()) / len(v.values()) for k, v in d.items()}
#
# print(image_score)


def quantification(tiles: list[str]):
    scores = []
    for s in range(1, 150):
        for file in tiles:
            tile_input = skimage.io.imread(file)
            count_foreground_pixel = np.count_nonzero(tile_input)
            grid_HSV = cv2.cvtColor(tile_input, cv2.COLOR_RGB2HSV)
            mask_1 = cv2.inRange(grid_HSV, (0, s, 50), (5, 255, 255))
            mask_2 = cv2.inRange(grid_HSV, (160, s, 20), (180, 255, 255))
            mask_red = cv2.bitwise_or(mask_1, mask_2)
            only_red = cv2.bitwise_and(tile_input, tile_input, mask=mask_red)

            count_treshold_red = np.count_nonzero(only_red)
            if count_treshold_red != 0:
                tile_score = (count_treshold_red / count_foreground_pixel) * 100
                scores.append(tile_score)
        total_score = sum(scores) / len(scores)
    return total_score


print(quantification(matching_files))

