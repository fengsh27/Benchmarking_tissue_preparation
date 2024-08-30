from typing import Tuple
import tensorflow
import matplotlib.pyplot as plt # for plotting images

from tifffile import tifffile

import numpy as np
import pandas as pd
import glob
from skimage.io import imsave
from skimage import img_as_ubyte
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
from tqdm import tqdm

from deepcell.applications import Mesmer # Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay

nuclear_markers = [0] # these are indices of the channels we want to use as the nuclear signal (only one here)
membrane_markers = [1, 4, 5, 6, 8, 14, 21, 22, 23, 24, 25, 27] # these are the indices we want to use as the membrane signal (referenced above)


def read_image(fn: str)->np.ndarray: # '/mnt/nfs/storage/Fusion_Registered_Report/Slide 1_20 min HIER 1h RT stain_Scan1.qptiff'
    img = glob.glob(fn)
    return tifffile.imread(img[0])

def generate_nuclear_and_membrane(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shp = (img.shape[1], img.shape[2])
    nuclear = np.zeros(shp)
    membrane = np.zeros(shp)

    for chn_index in range(len(img)):
        arr = img[chn_index, :, :] # slice the corresponding index in the first dimension
        if chn_index in nuclear_markers: # if the chn_index is in our list of nuclear markers, stack it to the nuclear array
            nuclear = np.add(nuclear, arr)
        elif chn_index in membrane_markers: # if the chn_index is a membrane_marker, stack it to the membrane array
            membrane = np.add(membrane, arr)
        else:
            pass

    return nuclear, membrane

def stack_nuclear_and_membrane(nuclear: np.ndarray, membrane: np.ndarray):
    # stack the nuclear and membrane arrays we created
    stack = np.stack((nuclear, membrane), axis=-1)
    # also expand to 4 dimensions
    return np.expand_dims(stack, 0)

def crop_out(
        img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int
    ) -> np.ndarray:
    return img[: ymin:ymax, xmin:xmax, :]

def run_segmentation(
    img: np.ndarray, 
    maxima_threshold: float, 
    interior_threshold: float
):
    image_mpp = 0.50
    maxima_threshold = 0.075
    interior_threshold = 0.2
    app = Mesmer()
    return app.predict(
        img, 
        image_mpp=image_mpp,
        postprocess_kwargs_whole_cell={
            "maxima_threshold": maxima_threshold,
            "interior_threshold": interior_threshold,
        },
    )

if __name__ == "__main__":
    img = read_image("/home/ubuntu/project/temp/Benchmarking_tissue_preparation_data/Slide 1_20 min HIER 1h RT stain_Scan1.qptiff")
    n, m = generate_nuclear_and_membrane(img)
    print(n)
    print(m)
