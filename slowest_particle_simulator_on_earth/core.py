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
    p_weights = np.zeros((nr_part, 3, 2))  # n × neighbour × xy contributions
    # Quadratic interpolation weights
    for i in range(nr_part):
        p = p_pos[i]  # particle coordinates
        cell_idx = np.floor(p)
        cell_diff = (p - cell_idx) - 0.5
        p_weights[i, 0, :] = 0.5 * np.power(0.5 - cell_diff, 2)
        p_weights[i, 1, :] = 0.75 - np.power(cell_diff, 2)
        p_weights[i, 2, :] = 0.5 * np.power(0.5 + cell_diff, 2)
    return p_weights


def particle_to_grid_volume(p_pos, p_mass, p_weights, cells):
    """Compute a volume field using particles."""
    dims = cells.shape[0], cells.shape[1]
    nr_part = p_pos.shape[0]

    p_volu = np.ones(nr_part)
    c_mass = np.ones(dims)

    for i in range(nr_part):
        p = p_pos[i, :]  # particle coordinates
        w = p_weights[i, :, :]  # particle neighbour interpolation weights
        m = p_mass[i]  # particle masses
        cell_idx = (np.floor(p)).astype(int)

        density = 0.
        for gx in range(3):
            for gy in range(3):
                weight = w[gx, 0] * w[gy, 1]

                cell = np.array([cell_idx[0] + gx - 1, cell_idx[1] + gy - 1])
                cell = cell.astype("int")
                density += c_mass[cell[0], cell[1]] * weight

        if density != 0:
            p_volu[i] = m / density

    return p_volu, c_mass


def particle_to_grid(p_pos, p_C, p_F, p_mass, p_velo, cells, p_weights, p_vals,
                     p_volu, c_mass, dt=1.0):
    """Compute a scalar field using particles."""
    # useful derivatives
    dims = cells.shape[0], cells.shape[1]
    nr_part = p_velo.shape[0]

    c_velo = np.zeros(dims + (2,))  # vector field
    c_values = np.zeros(dims)

    for i in range(nr_part):
        p = p_pos[i, :]  # particle coordinates

        # Deformation gradient??
        F = p_F[i, :, :]
        J = np.linalg.det(F)
        volume = p_volu[i] * J

        # Useful matrices for Neo-Hookean model
        F_T = np.copy(F.T)
        F_inv_T = np.linalg.inv(F_T)
        F_minus_F_inv_T = F - F_inv_T

        # MPM course equation 48
        elastic_lambda = 10.  # Parametrize this
        elastic_mu = 20.  # Parametrize this
        P_term_0 = elastic_mu * F_minus_F_inv_T
        P_term_1 = elastic_lambda * np.log(J) * F_inv_T
        P = P_term_0 + P_term_1

        # cauchy_stress = (1 / det(F)) * P * F_T
        # equation 38, MPM course
        stress = (1.0 / J) * np.matmul(P, F_T)

        # NOTE(nialltl): (M_p)^-1 = 4, see APIC paper and MPM course page 42
        # this term is used in MLS-MPM paper eq. 16. with quadratic weights,
        # Mp = (1/4) * (delta_x)^2. In this simulation, delta_x = 1, because
        # I scale the rendering of the domain rather than the domain itself.
        # We multiply by dt as part of the process of fusing the momentum and
        # force update for MLS-MPM
        eq_16_term_0 = -volume * 4 * stress * dt

        # 9 cell neighbourhood of the particle
        m = p_mass[i]  # particle masses
        v = p_velo[i, :]  # particle velocities
        w = p_weights[i, :, :]  # particle neighbour interpolation weights
        value = p_vals[i]  # particle values
        C = p_C[i, :, :]
        cell_idx = (np.floor(p)).astype(int)
        for gx in range(3):
            for gy in range(3):
                weight = w[gx, 0] * w[gy, 1]

                cell = np.array([cell_idx[0] + gx - 1, cell_idx[1] + gy - 1])
                cell = cell.astype("int")
                cell_dist = (cell - p) + 0.5
                Q = np.matmul(C, cell_dist)

                # MPM course equation 172
                mass_contrib = weight * m
                # c_mass[cell[0], cell[1]] += mass_contrib

                # APIC P2G momentum contribution
                c_velo[cell[0], cell[1], :] += mass_contrib * (v + Q)

                # Fused force/momentum update from MLS-MPM
                # see MLS-MPM paper, equation listed after eqn. 28
                momentum = np.matmul(eq_16_term_0 * weight, cell_dist)
                c_velo[cell[0], cell[1], :] += momentum

                # For carrying voxel values (grayscale image)
                value_contrib = weight * value
                c_values[cell[0], cell[1]] += value_contrib

                # NOTE: Cell velocity is actually momentum here. It will be
                # updated later.

    return c_velo, c_values


def grid_velocity_update(c_velo, c_mass, dt=1., gravity=0.05):
    """Operate on velocity grid."""
    idx = c_mass > 0
    # Convert momentum to velocity, apply gravity
    c_velo[idx, :] /= c_mass[idx, None]
    c_velo[idx, :] += dt * np.array([gravity, 0])

    # "slip" boundary conditions
    c_velo[0, :] = 0
    c_velo[-1, :] = 0
    c_velo[:, 0] = 0
    c_velo[:, -1] = 0

    return c_velo


def grid_to_particle_velocity(p_pos, p_velo, p_weights, p_C, p_F, c_velo,
                              dt=1., rule="clamp", bounce_factor=0.5):
    """Update particles based on velocities on the grid."""
    dims = c_velo.shape
    nr_part = p_pos.shape[0]
    # Reset particle velocity
    p_velo *= 0.

    for i in range(nr_part):
        p = p_pos[i, :]
        v = p_velo[i, :]
        w = p_weights[i, :, :]
        C = p_C[i, :, :]
        F = p_F[i, :, :]

        # NOTE(nialltl): Constructing affine per-particle momentum matrix from
        # APIC / MLS-MPM. See APIC paper:
        # <https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf>,
        # page 6 below eq. 11 for clarification. This is calculating C=B*(D^-1)
        # for APIC eq. 8, where B is calculated in the inner loop at (D^-1)=4
        # is a constant when using quadratic interpolation functions.
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

        C = B * 4
        # Advect particles
        p += v * dt

        # Act on escaped particles
        p, v = clamp(
            p, v, d_min_x=0, d_max_x=dims[0], d_min_y=0, d_max_y=dims[1],
            rule=rule, bounce_factor=bounce_factor)

        # Deformation gradient update - MPM course, equation 181
        # Fp' = (I + dt * p.C) * Fp
        F_new = np.eye(2)
        F_new += dt * C

        p_F[i, :, :] = np.matmul(F_new, F)
        p_C[i, :, :] = C

        # Update particles
        p_pos[i] = p[:]
        p_velo[i] = v[:]

    return p_pos, p_velo, p_C, p_F


def clamp(p, v, d_min_x=0, d_max_x=100, d_min_y=0, d_max_y=100,
          rule="slip", bounce_factor=-0.5):
    """Prevent particles escaping grid."""
    if rule == "clamp":  # Clamp positions
        if p[0] < d_min_x + 1:
            p[0] = d_min_x + 1
        elif p[0] > d_max_x - 2:
            p[0] = d_max_x - 2
        if p[1] < d_min_y + 1:
            p[1] = d_min_y + 1
        elif p[1] > d_max_y - 2:
            p[1] = d_max_y - 2

    elif rule == "slip":
        if p[0] < d_min_x + 1:
            p[0] = d_min_x + 1
            v[0] = 0
        elif p[0] > d_max_x - 2:
            p[0] = d_max_x - 2
            v[0] = 0
        if p[1] < d_min_y + 1:
            p[1] = d_min_y + 1
            v[1] = 0
        elif p[1] > d_max_y - 2:
            p[1] = d_max_y - 2
            v[1] = 0

    elif rule == "bounce":
        if p[0] < d_min_x + 1:
            p[0] = d_min_x + 1
            v[0] /= bounce_factor
        elif p[0] > d_max_x - 2:
            p[0] = d_max_x - 2
            v[0] /= bounce_factor
        if p[1] < d_min_y + 1:
            p[1] = d_min_y + 1
            v[1] /= bounce_factor
        elif p[1] > d_max_y - 2:
            p[1] = d_max_y - 2
            v[1] /= bounce_factor

    return p, v


def particle_pos_to_grid(p_pos, p_mass, cells, p_weights, p_vals):
    """Only use particle positions to generate a new 2D grid."""
    # useful derivatives
    dims = cells.shape[0], cells.shape[1]
    nr_part = p_pos.shape[0]

    c_mass = np.zeros(dims)  # scalar field
    c_values = np.zeros(dims)

    for i in range(nr_part):
        p = p_pos[i, :]  # particle coordinates
        m = p_mass[i]  # particle masses
        w = p_weights[i, :, :]  # particle neighbour interpolation weights
        value = p_vals[i]  # particle values

        # 9 cell neighbourhood of the particle
        cell_idx = (np.floor(p)).astype(int)
        for gx in range(3):
            for gy in range(3):
                weight = w[gx, 0] * w[gy, 1]

                cell = np.array([cell_idx[0] + gx - 1, cell_idx[1] + gy - 1])
                cell = cell.astype("int")

                # MPM course equation 172
                mass_contrib = weight * m

                # Insert into grid
                c_mass[cell[0], cell[1]] += mass_contrib

                # For carrying voxel values (grayscale image)
                value_contrib = weight * value
                c_values[cell[0], cell[1]] += value_contrib

    return c_mass, c_values
