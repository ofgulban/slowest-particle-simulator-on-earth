"""Material point method (2D) implementation in Python.

Reference
---------
https://nialltl.neocities.org/articles/mpm_guide.html
"""

import numpy as np


def particle_neighbours(p_pos):
    """Compute particle neighbouring grid cells."""
    dims = p_pos.shape
    p_neigh = np.zeros((dims[0], 4, 2), dtype="int")  # n x nr_neigh x coords
    for i in range(dims[0]):
        # Particle coordinates
        idx_x, idx_y = p_pos[i]

        # Neighbouring indices
        n_x = np.floor(idx_x)
        n_y = np.floor(idx_y)

        p_neigh[i, 0, :] = n_x, n_y
        p_neigh[i, 1, :] = n_x+1, n_y
        p_neigh[i, 2, :] = n_x, n_y+1
        p_neigh[i, 3, :] = n_x+1, n_y+1

    return p_neigh.astype(int)


def compute_interpolation_weights(p_pos):
    """Compute interpolation weights using particles on a grid."""
    # Useful derivatives
    nr_part = p_pos.shape[0]
    # Initialize new variables
    p_weights = np.zeros((nr_part, 3, 2))  # n × neighbour × xy contributios
    # Quadratic interpolation weights
    for i in range(nr_part):
        p = p_pos[i]  # particle coordinates
        cell_idx = np.floor(p)
        cell_diff = (p - cell_idx) - 0.5
        p_weights[i, 0, :] = 0.5 * np.power(0.5 - cell_diff, 2)
        p_weights[i, 1, :] = 0.75 - np.power(cell_diff, 2)
        p_weights[i, 2, :] = 0.5 * np.power(0.5 + cell_diff, 2)
    return p_weights


def particle_to_grid(p_pos, p_C, p_mass, p_velo, cells, p_weights, p_vals):
    """Compute a scalar field using particles."""
    # useful derivatives
    dims = cells.shape[0], cells.shape[1]
    nr_part = p_velo.shape[0]

    c_mass = np.zeros(dims)  # scalar field
    c_velo = np.zeros(dims + (2,))  # vector field
    c_values = np.zeros(dims)

    for i in range(nr_part):
        p = p_pos[i, :]  # particle coordinates
        C = p_C[i, :, :]  # TODO: What is this variable?
        m = p_mass[i]  # particle masses
        v = p_velo[i, :]  # particle velocities
        w = p_weights[i, :, :]  # particle neighbour interpolation weights
        value = p_vals[i]  # particle values

        # 9 cell neighbourhood of the particle
        cell_idx = (np.floor(p)).astype(int)
        for gx in range(3):
            for gy in range(3):
                weight = w[gx, 0] * w[gy, 1]

                cell = np.array([cell_idx[0] + gx - 1, cell_idx[1] + gy - 1])
                cell = cell.astype("int")
                cell_dist = (cell - p) + 0.5
                Q = np.dot(C, cell_dist)

                # MPM course equation 172
                mass_contrib = weight * m

                # Insert into grid
                c_mass[cell[0], cell[1]] += mass_contrib
                c_velo[cell[0], cell[1], :] += mass_contrib * (v + Q)

                # For carrying voxel values (grayscale image)
                value_contrib = weight * value
                c_values[cell[0], cell[1]] += value_contrib

                # NOTE: Cell velocity is actually momentum here. It will be
                # updated later.

    return c_mass, c_velo, c_values


def grid_velocity_update(c_velo, c_mass, dt=1., gravity=0.05):
    """Operate on velocity grid."""
    idx = c_mass > 0
    c_velo[idx, :] /= c_mass[idx, None]
    c_velo[idx, :] += dt * np.array([gravity, 0])
    return c_velo


def grid_to_particle_velocity(p_pos, p_velo, p_weights, c_velo, dt=1.,
                              rule="bounce", bounce_factor=0.5):
    """Update particles based on velocities on the grid."""
    dims = c_velo.shape
    nr_part = p_pos.shape[0]
    # Reset particle velocity
    p_velo *= 0

    for i in range(nr_part):
        p = p_pos[i, :]
        v = p_velo[i, :]
        w = p_weights[i, :, :]

        # Construct affine per-particle momentum matrix from (APIC)/MLS-MPM.
        B = np.zeros((2, 2))

        # 9 cell neighbourhood of the particle
        cell_idx = (np.floor(p)).astype(int)
        for gx in range(3):
            for gy in range(3):
                weight = w[gx, 0] * w[gy, 1]

                cell = np.floor([cell_idx[0] + gx - 1, cell_idx[1] + gy - 1])
                cell = cell.astype("int")
                cell_dist = (cell - p) + 0.5
                weighted_velocity = c_velo[cell_idx[0], cell_idx[1]] * weight

                # APIC paper equation 10, constructing inner term for B
                term = np.array([weighted_velocity * cell_dist[0],
                                 weighted_velocity * cell_dist[1]])

                B += term
                v += weighted_velocity

        # p_C[i] = B * 4  # unused for now

        # Advect particles
        p += v * dt

        # Act on escaped particles
        p, v = clamp(p, v, d_min=0, d_max=dims[1], rule="bounce",
                     bounce_factor=bounce_factor)

        # Update particles
        p_pos[i] = p[:]
        p_velo[i] = v[:]

    return p_pos, p_velo


def clamp(p, v, d_min=0, d_max=100, rule="slip", bounce_factor=-0.5):
    """Prevent particles escaping grid."""

    if rule == "slip":  # Clamp positions
        if p[0] < d_min + 1:
            p[0] = d_min + 1
        elif p[0] > d_max - 2:
            p[0] = d_max - 2
        if p[1] < d_min + 1:
            p[1] = d_min + 1
        elif p[1] > d_max - 2:
            p[1] = d_max - 2

    elif rule == "bounce":
        if p[0] < d_min + 1:
            p[0] = d_min + 1
            v[0] /= bounce_factor
        elif p[0] > d_max - 2:
            p[0] = d_max - 2
            v[0] /= bounce_factor
        if p[1] < d_min + 1:
            p[1] = d_min + 1
            v[1] /= bounce_factor
        elif p[1] > d_max - 2:
            p[1] = d_max - 2
            v[1] /= bounce_factor

    return p, v
