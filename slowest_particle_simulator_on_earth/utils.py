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
