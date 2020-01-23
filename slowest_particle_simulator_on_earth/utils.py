"""Utility functions and procedures."""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, dirname, isdir


def save_img(img, out_dir, suffix="", invert=False):
    """Save a grid scalar field as png file."""
    img_out = np.copy(img)
    img_out[img < 0] = 0
    img_out[img > 1] = 1
    img_out = np.repeat(img_out[..., None], 3, axis=2)  # For greyscale RGB

    if invert:
        img_out *= -1
        img_out += 1

    out_name = "frame_{}.png".format(suffix)
    out_path = join(out_dir, out_name)
    plt.imsave(out_path, img_out, origin="upper")


def create_export_folder(nii_filename):
    """Create export folder based on filename."""
    i = 0
    nii_filedir = dirname(nii_filename)
    output_dir = join(nii_filedir, "export_" + str(i).zfill(2))
    while isdir(output_dir):
        i += 1
        output_dir = join(nii_filedir, "export_" + str(i).zfill(2))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def embed_data_into_square_lattice(data):
    """Insert MR image into square 2D array."""
    dims = np.array(data.shape)

    offset_x = int((dims.max() - dims[0]) / 2.)
    offset_y = int((dims.max() - dims[1]) / 2.)

    temp = np.zeros((dims.max(), dims.max()))
    temp[offset_x:offset_x+dims[0], offset_y:offset_y+dims[1]] = data
    return temp


def normalize_data_range(data, thr_min=0, thr_max=500):
    """Negative numbers become 0 and normalize non-zero data into 0-1 range."""
    data -= thr_min
    data[data < 0] = 0
    data[data > (thr_max - thr_min)] = thr_max - thr_min
    data = data / (thr_max - thr_min)
    data *= 0.5
    return data
