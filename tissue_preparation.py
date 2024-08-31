from typing import Tuple
import os
import tensorflow
import matplotlib.pyplot as plt # for plotting images

from tifffile import tifffile

import numpy as np
import pandas as pd
import glob
from skimage.io import imsave, imread
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
    if ymax - ymin != 5500:
        raise ValueError(f"{yamx} - {ymin} is not 5500")
    if xmax - xmin != 7000:
        raise ValueError(f"{xamx} - {xmin} is not 7000")
    
    print(ymin)
    print(ymax)
    print(xmin)
    print(xmax)
    return img[:, ymin:ymax, xmin:xmax, :]

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

def extract_sc_features(
        mesmer_result_fn: str,
        img: np.ndarray, 
        xmin: int, ymin: int,
        xmax: int, ymax: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    markers = ['DAPI', 'CD3', 'aSMA', 'CD15', 'CD4', 'CD8', 'CD11b', 'CD11c', 'CD20', 'CD21', 'H2K27me3', 'Ki.67', 'HLA.DRA', 'Histone.H3', 'CD68', 'DC.SIGN', 'Foxp3', 'PD.1', 'CD163', 'H3K27ac', 'Granzyme.B', 'CD31', 'CD206', 'CD138', 'NaK.ATPase', 'CD45RA', 'CD45', 'Cytokeratin']

    # load in mask
    mask = imread(mesmer_result_fn)
    
    # transpose the stack dimensions
    array_list = np.transpose(img, (1, 2, 0))

    # crop the array to match cropping done earlier
    cropped_array_list = array_list[ymin:ymax, xmin:xmax, :]

    stats = skimage.measure.regionprops(mask)
    cell_count = len(stats) # number of actual cells not always equal to np.max(mask) 
    marker_count = len(markers)

    # empty containers of zeros
    data = np.zeros((cell_count, marker_count))
    dataScaleSize = np.zeros((cell_count, marker_count))
    cellSizes = np.zeros((cell_count, 1))
    cell_props = np.zeros((cell_count, 3))

    # extract info
    for i in tqdm(range(cell_count)): # tqdm creates the progress bar
        cellLabel = stats[i].label
        label_counts = [cropped_array_list[coord[0], coord[1], :] for coord in stats[i].coords] # all markers for this cell
        data[i, 0:marker_count] = np.sum(label_counts, axis = 0) # sum the marker expression for this cell
        dataScaleSize[i, 0:marker_count] = np.sum(label_counts, axis = 0) / stats[i].area # scale the sum by size
        cellSizes[i] = stats[i].area # cell size
        cell_props[i, 0] = cellLabel
        cell_props[i, 1] = stats[i].centroid[0] # Y centroid
        cell_props[i, 2] = stats[i].centroid[1] # X centroid

    data_df = pd.DataFrame(data)
    data_df.columns = markers
    data_full = pd.concat((pd.DataFrame(cell_props, columns = ["cellLabel", "Y_cent", "X_cent"]), pd.DataFrame(cellSizes, columns = ["cellSize"]), data_df), axis=1)

    dataScaleSize_df = pd.DataFrame(dataScaleSize)
    dataScaleSize_df.columns = markers
    dataScaleSize_full = pd.concat((pd.DataFrame(cell_props, columns = ["cellLabel", "Y_cent", "X_cent"]), pd.DataFrame(cellSizes, columns = ["cellSize"]), dataScaleSize_df), axis = 1)
    
    return data_full, dataScaleSize_full
    # save the dataframes
    # data_full.to_csv(os.path.join(out, 'data_slide1.csv'), index = False)
    # dataScaleSize_full.to_csv(os.path.join(out, 'dataScaleSize_slide1.csv'), index = False)

if __name__ == "__main__":
    img = read_image("/home/ubuntu/project/temp/Benchmarking_tissue_preparation_data/Slide 1_20 min HIER 1h RT stain_Scan1.qptiff")
    n, m = generate_nuclear_and_membrane(img)
    print(n)
    print(m)
