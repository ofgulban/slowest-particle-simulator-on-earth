"""Utility functions and procedures."""

import matplotlib.pyplot as plt
import numpy as np
from os.path import join


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
