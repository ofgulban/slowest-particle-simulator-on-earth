"""Script example 1: skull explodes but brain stays still."""

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
NII_FILE = "/home/faruk/Git/slowest-particle-simulator-on-earth/script_examples/sample_data/sample_T1w_cropped.nii.gz"
OUT_DIR = create_export_folder(NII_FILE)
MASK = "/home/faruk/Git/slowest-particle-simulator-on-earth/script_examples/sample_data/sample_T1w_cropped_brain.nii.gz"

SLICE_NR = 3

DIMS = (256, 256)
NR_ITER = 200
DT = 1  # Time step (smaller = more accurate simulation)
GRAVITY = 0.05

THR_MIN = 200
THR_MAX = 500

OFFSET_X = 0
OFFSET_Y = 32

# =============================================================================
# Load nifti
nii = nb.load(NII_FILE)
data = nii.get_fdata()[:, SLICE_NR, :]
data = embed_data_into_square_lattice(data)
data = normalize_data_range(data, thr_min=THR_MIN, thr_max=THR_MAX)

# Load Mask
mask = nb.load(MASK)
mask = mask.get_fdata()[:, SLICE_NR, :]
mask = embed_data_into_square_lattice(mask)
idx_mask_x, idx_mask_y = np.where((mask+1) % 2)

# =============================================================================
# Initialize particles
x, y = np.where(data * ((mask+1) % 2))
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
p_velo[:, 0] = (np.random.rand(NR_PART) + 0) * -1
p_velo[:, 1] = (np.random.rand(NR_PART) - 0.5) * 4

p_mass = np.ones(NR_PART)

p_C = np.zeros((NR_PART, 2, 2))

# Initialize cells
cells = np.zeros(data.shape)

# =============================================================================
# Start simulation iterations
for t in range(NR_ITER):
    p_weights = compute_interpolation_weights(p_pos)

    c_mass, c_velo, c_values = particle_to_grid(
        p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals)

    c_velo = grid_velocity_update(
        c_velo, c_mass, dt=DT, gravity=GRAVITY)

    # Manipulate grid velocities
    c_velo[idx_mask_x, idx_mask_y, :] *= -1.25

    p_pos, p_velo = grid_to_particle_velocity(
        p_pos, p_velo, p_weights, c_velo, dt=DT,
        rule="bounce", bounce_factor=-1.25)

    # Add static
    c_values[idx_mask_x, idx_mask_y] += data[idx_mask_x, idx_mask_y]

    # Adjust brightness w.r.t. mass
    c_values[c_mass > 2] /= c_mass[c_mass > 2]
    save_img(c_values, OUT_DIR, suffix=str(t+1).zfill(3))
    print("Iteration: {}".format(t))
