"""Main entry point."""

import argparse
import numpy as np
import slowest_particle_simulator_on_earth.config as cfg
from slowest_particle_simulator_on_earth.core import (
    compute_interpolation_weights, particle_to_grid, grid_velocity_update,
    grid_to_particle_velocity)
from slowest_particle_simulator_on_earth.utils import (
    nifti_reader, save_img, create_export_folder, embed_data_into_square_lattice,
    normalize_data_range, log_welcome, log_progress)


def main():
    """Commandline interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'filename',  metavar='path',
        help="Path to nifti file. Only positive values will be visualized \
        Use a masked image for faster iterations."
        )
    parser.add_argument(
        '--iterations', type=int, required=False,
        metavar=cfg.iterations, default=cfg.iterations,
        help="Number of iterations. Equal to number of frames generated."
        )
    parser.add_argument(
        '--slice_axis', type=int, required=False,
        metavar=cfg.slice_axis, default=cfg.slice_axis,
        help="Slice axis 0, 1, 2 (e.g. transversal, coronal, saggital)"
        )
    parser.add_argument(
        '--slice_number', type=int, required=False,
        metavar=cfg.slice_number, default=cfg.slice_number,
        help="Slice on Y axis that will be visualized."
        )
    parser.add_argument(
        '--thr_min', type=int, required=False,
        metavar=cfg.thr_min, default=cfg.thr_min,
        help="Change values below this threshold to zero."
        )
    parser.add_argument(
        '--thr_max', type=int, required=False,
        metavar=cfg.thr_max, default=cfg.thr_max,
        help="Change values above this threshold to this value."
        )
    parser.add_argument(
        '--rotate', type=int, required=False,
        metavar=cfg.degree_rotate, default=cfg.degree_rotate,
        help="Rotate the image by 90, 180, 270 degree by the slice axis."
        )

    args = parser.parse_args()
    cfg.iterations = args.iterations
    cfg.slice_axis = args.slice_axis
    cfg.slice_number = args.slice_number
    cfg.thr_min = args.thr_min
    cfg.thr_max = args.thr_max
    cfg.degree_rotate = args.rotate

    log_welcome()

    # =========================================================================
    # Parameters
    NII_FILE = args.filename
    NR_ITER = cfg.iterations
    SLICE_AXIS = cfg.slice_axis
    SLICE_NR = cfg.slice_number

    OUT_DIR = create_export_folder(NII_FILE)

    # Parameters that can be added to CLI in the future
    DT = 1  # Time step (smaller = more accurate simulation)
    GRAVITY = 0.05

    # -------------------------------------------------------------------------
    # Load nifti
    data = nifti_reader(NII_FILE, SLICE_AXIS, SLICE_NR, cfg.degree_rotate)
    data = embed_data_into_square_lattice(data)
    data = normalize_data_range(data, thr_min=cfg.thr_min, thr_max=cfg.thr_max)

    # -------------------------------------------------------------------------
    # Initialize particles
    x, y = np.where(data)
    p_pos = np.stack((x, y), axis=1)
    p_pos = p_pos.astype(float)

    # Record voxel values into particles
    p_vals = data[x, y]
    x, y = None, None

    # Move particles to the center of cells
    p_pos[:, 0] += 0.5
    p_pos[:, 1] += 0.5

    NR_PART = p_pos.shape[0]

    p_velo = np.zeros((NR_PART, 2))
    p_velo[:, 0] = (np.random.rand(NR_PART) + 1.5) * -1
    p_velo[:, 1] = (np.random.rand(NR_PART) - 0.5) * 4

    p_mass = np.ones(NR_PART)

    p_C = np.zeros((NR_PART, 2, 2))

    # Initialize cells
    cells = np.zeros(data.shape)

    # Some informative prints
    print("  Output folder:\n  {}".format(OUT_DIR))
    print("  Number of particles: {}".format(NR_PART))

    # -------------------------------------------------------------------------
    # Start simulation iterations
    for t in range(NR_ITER):
        p_weights = compute_interpolation_weights(p_pos)

        c_mass, c_velo, c_values = particle_to_grid(
            p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals)

        c_velo = grid_velocity_update(
            c_velo, c_mass, dt=DT, gravity=GRAVITY)

        p_pos, p_velo = grid_to_particle_velocity(
            p_pos, p_velo, p_weights, c_velo, dt=DT,
            rule="bounce", bounce_factor=-1.25)

        # Adjust brightness w.r.t. mass
        c_values[c_mass > 2] /= c_mass[c_mass > 2]
        save_img(c_values, OUT_DIR, suffix=str(t+1).zfill(3))
        log_progress

    # =========================================================================


if __name__ == "__main__":
    main()
    print('Finished.')
