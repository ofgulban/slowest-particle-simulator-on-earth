"""Script example 5: read png."""

import numpy as np
import cv2
from scipy.ndimage import zoom
from slowest_particle_simulator_on_earth.core import (
    compute_interpolation_weights, particle_to_grid, grid_velocity_update,
    grid_to_particle_velocity)
from slowest_particle_simulator_on_earth.utils import (
    save_img, create_export_folder, embed_data_into_square_lattice,
    normalize_data_range, log_welcome, log_progress)


# =============================================================================
# Parameters
PNG_FILE1 = "/home/faruk/gdrive/test_brainsplode2/ohbm/OHBM2020_1.png"
PNG_FILE2 = "/home/faruk/gdrive/test_brainsplode2/ohbm/g982.png"
OUT_DIR = create_export_folder(PNG_FILE1)

NR_ITER = 200

DT = 1  # Time step (smaller = more accurate simulation)
GRAVITY = 0.2

# =============================================================================
# Load png (BGR)
data_in_1 = cv2.imread(PNG_FILE1)
data_in_1 = np.asarray(data_in_1, dtype=float)
data_in_2 = cv2.imread(PNG_FILE2)
data_in_2 = np.asarray(data_in_2, dtype=float)

# Select a channel
data_1_moving = data_in_1[:, :, 2]
data_1_static = data_in_1[:, :, 1]
data_1_bounce = data_in_1[:, :, 0]

# Downsample
data_1_moving = zoom(data_1_moving, 1/2, mode="nearest", prefilter=False)
data_1_static = zoom(data_1_static, 1/2, mode="nearest", prefilter=False)
data_1_bounce = zoom(data_1_bounce, 1/2, mode="nearest", prefilter=False)

# Normalize to 0-0.5 range
data_1_moving = normalize_data_range(data_1_moving, thr_min=0, thr_max=255)*2

static = np.where(data_1_static)

# Binarize
# data = ~(data > 0)
# Null edges
# data[-1, :] = 0
# data[:, -1] = 0

# =============================================================================
# Initialize particles
x, y = np.where(data_1_moving)
p_pos = np.stack((x, y), axis=1)
p_pos = p_pos.astype(float)

NR_PART = p_pos.shape[0]

# Record voxel values into particles
p_vals = data_1_moving[x, y]
x, y = None, None

# Bounce voxels
idx_mask_x, idx_mask_y = np.where(data_1_bounce != 0)

# Move particles to the center of cells
p_pos[:, 0] += 0.5
p_pos[:, 1] += 0.5

p_velo = np.zeros((NR_PART, 2))
p_velo[:, 0] = (np.random.rand(NR_PART) + 0.25) * -1
p_velo[:, 1] = (np.random.rand(NR_PART) - 0.5) * 1

p_mass = np.ones(NR_PART)

p_C = np.zeros((NR_PART, 2, 2))

# Initialize cells
cells = np.zeros(data_1_moving.shape)

# Some informative prints
log_welcome()
print("Output folder: {}".format(OUT_DIR))
print("Number of particles: {}".format(NR_PART))

# =============================================================================
# Start simulation iterations
for t in range(NR_ITER):

    # if t == 40:
    #     p_velo[:, 0] = (np.random.rand(NR_PART) -0.5) * -10

    p_weights = compute_interpolation_weights(p_pos)

    c_mass, c_velo, c_values = particle_to_grid(
        p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals)

    # Add static
    c_values[static] = 1 - c_values[static]

    c_velo = grid_velocity_update(
        c_velo, c_mass, dt=DT, gravity=GRAVITY)

    # Manipulate grid velocities
    c_velo[idx_mask_x, idx_mask_y, :] *= -0.75

    p_pos, p_velo = grid_to_particle_velocity(
        p_pos, p_velo, p_weights, c_velo, dt=DT,
        rule="bounce", bounce_factor=-1.5)

    # Adjust brightness w.r.t. mass
    c_values[c_mass > 2] /= c_mass[c_mass > 2]
    save_img(c_values, OUT_DIR, suffix=str(t+1).zfill(3))
    log_progress(t, NR_ITER)
