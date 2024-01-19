import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

#%% Settings
input_dir = "./images"
pattern = "*.png"  # pattern to match filenames
sat = 50 # saturation threshold 

#%% Prep
matching_files = glob.glob(os.path.join(input_dir, pattern)) # list all files in the folder that match the pattern

#%% Quantification
def plot_result(img, mask, score):
    fig, axs = plt.subplots(1, 2, figsize = (7, 7))
    
    axs[0].imshow(img[...,::-1]) # BGR to RGB (needed by plt)
    axs[0].set_title('Original')
    
    # Change color of pixels classified as fibrosis
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    img_rgb = np.repeat(img_gray[:, :, np.newaxis], 3, axis=2)
    img_rgb[mask == 255, :] = (102, 85, 92) # color of pixels quantified

    axs[1].imshow(img_rgb)
    axs[1].set_title("Quantified: {}%".format(score))

    [ax.axis('off') for ax in axs]
    plt.tight_layout()
    

def quantification(file, plot = False):
    """Quantify amount of fibrosis in a single image"""
    img = cv2.imread(file) # loaded as B G R
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count_foreground_pixel = np.count_nonzero(img_gray)
    grid_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(grid_HSV, (0, sat, 50), (5, 255, 255)) # (low_H, low_S, low_V), (high_H, high_S, high_V)
    mask_2 = cv2.inRange(grid_HSV, (160, sat, 20), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_1, mask_2)

    if count_foreground_pixel != 0:
        score = round((np.count_nonzero(mask_red) / count_foreground_pixel) * 100, 1)
        quant = np.where(mask_red == 255)
        idx = np.vstack((quant[1], quant[0])).T # 1 - col - x; 0 - row - y
    else:
        score = np.nan
        idx = np.nan
    
    if plot:
        plot_result(img, mask_red, score)
                
    return score, idx


def quantify(tiles: list[str]):
    """Quantify amount of fibrosis in a set of images"""
    scores = []
    pixels = dict()
    for tile in tiles:
        tile_score, xy = quantification(tile) 
        scores.append(tile_score)
        pixels[tile] = xy
        
    tiles_score = np.nanmean(scores)
    
    return round(tiles_score), pixels


total_score, pixels_dict = quantify(matching_files)
