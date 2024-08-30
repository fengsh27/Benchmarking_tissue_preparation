
import matplotlib.pyplot as plt
from deepcell.utils.plot_utils import (
    create_rgb_image,
    make_outline_overlay,
)
import skimage
from skimage.io import imsave
from skimage import img_as_ubyte

from .tissue_preparation import (
    read_image, 
    generate_nuclear_and_membrane,
    crop_out,
    stack_nuclear_and_membrane,
    run_segmentation,
)

def test_generate_nuclear_and_membrane():
    data_dir = "/home/ubuntu/project/temp/Benchmarking_tissue_preparation_data/"
    out_dir = data_dir + "out/"

    img = read_image(f"{data_dir}Slide 1_20 min HIER 1h RT stain_Scan1.qptiff")
    n, m = generate_nuclear_and_membrane(img)
    stacked_img = stack_nuclear_and_membrane(n, m)

    cropped_img = crop_out(stacked_img, 5000, 12000, 9500, 15000)
    predictions = run_segmentation(cropped_img)

    rgb_img = create_rgb_image(cropped_img)
    overlay = make_outline_overlay(rgb_data=rgb_img, predictions=predictions)
    imsave(f"{out_dir}seg_outline.tiff", img_as_ubyte(overlay[0, ..., 0], check_contrast=False))
    imsave(f"{out_dir}seg_overlay.tiff", img_as_ubyte(overlay[0, ...], check_contrast=False))
    imsave(f"{out_dir}MESMER_mask.tiff", predictions[0, ..., 0], check_contrast=False)


    