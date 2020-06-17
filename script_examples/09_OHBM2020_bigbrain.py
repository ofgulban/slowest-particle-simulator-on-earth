"""OHBM2020 BrainHack."""

import nibabel as nb
import numpy as np
from slowest_particle_simulator_on_earth.core import (
    compute_interpolation_weights, particle_to_grid, grid_velocity_update,
    grid_to_particle_velocity)
from slowest_particle_simulator_on_earth.utils import (
    save_img, create_export_folder, embed_data_into_square_lattice,
    normalize_data_range)

# =============================================================================
# Parameters
NII_FILE = "/home/faruk/gdrive/OHBM2020_hackathon/bigbrain/full16_100um_optbal_roi_250um.nii.gz"
MASK = "/home/faruk/gdrive/OHBM2020_hackathon/bigbrain/cortex_fromKonrad_roi_250um_layers_equidist.nii.gz"

OUT_DIR = create_export_folder(NII_FILE)
print("Output folder: {}".format(OUT_DIR))

SLICE_NR = 145

DIMS = (256, 256)
NR_ITER = 400
DT = 1  # Time step (smaller = more accurate simulation)
GRAVITY = 0.05

THR_MIN = 35000
THR_MAX = 65000

# =============================================================================
# Load nifti
nii = nb.load(NII_FILE)
data = nii.get_fdata()[:, SLICE_NR, :]
data = embed_data_into_square_lattice(data)
data = normalize_data_range(data, thr_min=THR_MIN, thr_max=THR_MAX)

# Load Mask
mask = nb.load(MASK)
mask = mask.get_fdata()[:, SLICE_NR, :]
mask = mask.astype(int)
uniq = np.unique(mask)[::-1]
mask = embed_data_into_square_lattice(mask)

# (optional) rotate
data = data[::-1, ::-1]
mask = mask[::-1, ::-1]

# Initialize cells
cells = np.zeros(data.shape)

# =============================================================================
p_pos = np.array([[], []]).T
p_vals = np.array([])
p_velo = np.array([[], []]).T
p_mass = np.array([])
t_offset = 0
stage = 0
for i in uniq[0:-1]:  # Big numbers explode first
    # Note coordinates of voxels in each stage (0 is for static)
    x, y = np.where(mask == i)
    p_pos_new = np.stack((x, y), axis=1)
    p_pos_new = p_pos_new.astype(float)
    # Move particles to the center of cells
    p_pos_new[:, 0] += 0.5
    p_pos_new[:, 1] += 0.5
    # Add new particles to all particles
    p_pos = np.concatenate((p_pos, p_pos_new), axis=0)
    nr_new_part = p_pos_new.shape[0]
    nr_part = p_pos.shape[0]

    # Record voxel values into particles
    p_vals_new = data[x, y]
    p_vals = np.concatenate((p_vals, p_vals_new), axis=0)
    x, y = None, None

    p_velo_new = np.zeros((nr_new_part, 2))
    p_velo_new[:, 0] = (np.random.rand(nr_new_part) + 0.75) * -1
    p_velo_new[:, 1] = (np.random.rand(nr_new_part) - 0.5) * 4
    p_velo = np.concatenate((p_velo, p_velo_new), axis=0)

    p_mass_new = np.ones(p_pos_new.shape[0])
    p_mass = np.concatenate((p_mass, p_mass_new), axis=0)

    p_C = np.zeros((nr_part, 2, 2))

    # Some informative prints
    print("Stage: {}".format(stage))
    print("Number of particles: {}".format(nr_part))

    # Static voxels
    idx_mask_x, idx_mask_y = np.where(mask < i)

    # =============================================================================
    # Save initial image
    temp_brigtness_multiplier = 1.5
    temp = data * temp_brigtness_multiplier
    save_img(temp, OUT_DIR, suffix=str(0).zfill(3))

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

        # Add static
        c_values[idx_mask_x, idx_mask_y] += data[idx_mask_x, idx_mask_y]

        # Adjust brightness w.r.t. mass
        c_values[c_mass > 2] /= c_mass[c_mass > 2]
        c_values *= temp_brigtness_multiplier
        save_img(c_values, OUT_DIR, suffix=str(t_offset+t+1).zfill(3))
        print("  Iteration: {}".format(t_offset+t))

    t_offset += t + 1
    stage += 1
