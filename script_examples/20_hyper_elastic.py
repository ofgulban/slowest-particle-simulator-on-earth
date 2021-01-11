"""Script example 2: brain explodes skull stays still."""

import nibabel as nb
import numpy as np
from slowest_particle_simulator_on_earth.core import (
    compute_interpolation_weights, particle_to_grid, grid_velocity_update,
    grid_to_particle_velocity, particle_to_grid_volume)
from slowest_particle_simulator_on_earth.utils import (
    save_img, create_export_folder, embed_data_into_square_lattice,
    normalize_data_range)

# =============================================================================
# Parameters
NII_FILE = "/home/faruk/Git/slowest-particle-simulator-on-earth/script_examples/sample_data/sample_T1w_cropped_brain_4x.nii.gz"
OUT_DIR = create_export_folder(NII_FILE)

SLICE_NR = 3
NR_ITER = 200

DT = 1.  # Time step (smaller = more accurate simulation)
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

# Mask
idx_mask_x, idx_mask_y = np.where(data == 0)

# =============================================================================
# Initialize particles
x, y = np.where(data * (data != 0))
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
p_velo[:, 0] = (np.random.rand(NR_PART) + 0) * -0.5
p_velo[:, 1] = (np.random.rand(NR_PART) - 0.5) * 0.5

p_mass = np.ones(NR_PART)

p_C = np.zeros([2, 2])  # Deformation gradient matrix
p_C = np.repeat(p_C[np.newaxis, :, :], NR_PART, axis=0)
p_F = np.eye(2)  # Deformation gradient matrix
p_F = np.repeat(p_F[np.newaxis, :, :], NR_PART, axis=0)

# Initialize cells
cells = np.zeros(data.shape)

# Some informative prints
print("Output folder: {}".format(OUT_DIR))
print("Number of particles: {}".format(NR_PART))

# =============================================================================
# Start simulation iterations
for t in range(NR_ITER):
    p_weights = compute_interpolation_weights(p_pos)

    p_volu, c_mass = particle_to_grid_volume(p_pos, p_mass, p_weights, cells)

    c_velo, c_values = particle_to_grid(
        p_pos, p_C, p_F, p_mass, p_velo, cells, p_weights, p_vals, p_volu,
        dt=DT)

    c_velo = grid_velocity_update(
        c_velo, c_mass, dt=DT, gravity=GRAVITY)

    p_pos, p_velo, p_C, p_F = grid_to_particle_velocity(
        p_pos, p_velo, p_weights, p_C, p_F, c_velo, dt=DT,
        rule="slip", bounce_factor=-0.9)

    # Adjust brightness w.r.t. mass
    c_values[c_mass > 2] /= c_mass[c_mass > 2]
    save_img(c_values, OUT_DIR, suffix=str(t+1).zfill(3))
    print("Iteration: {}".format(t))
