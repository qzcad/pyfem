#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_mindlin_plate_laminated
    from assembly2d import assembly_initial_value
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    from force import thermal_force_plate_5
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import lil_matrix
    from numpy import array, zeros
    from quadrature import legendre_quad

    a = 10.0 # A side of a square plate
    h = a / 100.0 # A thickness of a square plate
    e = 1.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    alpha = 1.0E-6
    n = 101
    freedom = 5
    d = plane_stress_isotropic(e, nu)
    (nodes, elements) = rectangular_quads(x_count=n, y_count=n, x_origin=0.0, y_origin=0., width=a, height=a)

    stiffness = assembly_quads_mindlin_plate_laminated(nodes, elements, [h], [d])

    print("Evaluating force...")

    force = thermal_force_plate_5(nodes=nodes, elements=elements, thicknesses=[h], elasticity_matrices=[d], alpha_t=alpha)

    # print("Evaluating boundary conditions...")
    # for i in range(len(nodes)):
    #     if (abs(nodes[i, 0] - 0) < 0.0000001) or (abs(nodes[i, 0] - a) < 0.0000001):
    #         assembly_initial_value(stiffness, force, freedom * i, 0.0)
    #         assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)
    #         assembly_initial_value(stiffness, force, freedom * i + 2, 0.0)
    #         # assembly_initial_value(stiffness, force, freedom * i + 3, 0.0)
    #         # assembly_initial_value(stiffness, force, freedom * i + 4, 0.0)
    #     if (abs(nodes[i, 1] - 0) < 0.0000001) or (abs(nodes[i, 1] - a) < 0.0000001):
    #         assembly_initial_value(stiffness, force, freedom * i, 0.0)
    #         assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)
    #         assembly_initial_value(stiffness, force, freedom * i + 2, 0.0)
            # assembly_initial_value(stiffness, force, freedom * i + 3, 0.0)
            # assembly_initial_value(stiffness, force, freedom * i + 4, 0.0)

    print("Solving a system of linear equations")

    x = spsolve(stiffness, force)

    xi = array([-1.0, 1.0, 1.0, -1.0])
    eta = array([-1.0, -1.0, 1.0, 1.0])
    nodes_count = len(nodes)
    sigma_x = zeros(nodes_count)
    sigma_y = zeros(nodes_count)
    tau_xy = zeros(nodes_count)
    adjacent = zeros(nodes_count)
    # mises = zeros(nodes_count)
    for element in elements:
        vertices = nodes[element[:], :]
        displacement = zeros(freedom * 4)
        for j in range(freedom):
            displacement[j::freedom] = x[element[:] * freedom + j]

        for i in range(4):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            bm = array([
                [shape_dx[0], 0.0, 0.0, 0.0, 0.0, shape_dx[1], 0.0, 0.0, 0.0, 0.0, shape_dx[2], 0.0, 0.0, 0.0, 0.0,
                 shape_dx[3], 0.0, 0.0, 0.0, 0.0],
                [0.0, shape_dy[0], 0.0, 0.0, 0.0, 0.0, shape_dy[1], 0.0, 0.0, 0.0, 0.0, shape_dy[2], 0.0, 0.0, 0.0, 0.0,
                 shape_dy[3], 0.0, 0.0, 0.0],
                [shape_dy[0], shape_dx[0], 0.0, 0.0, 0.0, shape_dy[1], shape_dx[1], 0.0, 0.0, 0.0, shape_dy[2],
                 shape_dx[2], 0.0, 0.0, 0.0, shape_dy[3], shape_dx[3], 0.0, 0.0, 0.0]
            ])
            sigma = d.dot(bm).dot(displacement)
            sigma_x[element[i]] += sigma[0]
            sigma_y[element[i]] += sigma[1]
            tau_xy[element[i]] += sigma[2]
            adjacent[element[i]] += 1.0

    sigma_x /= adjacent
    sigma_y /= adjacent
    tau_xy /= adjacent
    u = x[0::freedom]
    v = x[1::freedom]
    w = x[2::freedom]
    theta_x = x[3::freedom]
    theta_y = x[4::freedom]
    print(min(u), " <= u <= ", max(u))
    print(min(v), " <= v <= ", max(v))
    print(min(w), " <= w <= ", max(w))
    print(min(theta_x), " <= theta x <= ", max(theta_x))
    print(min(theta_y), " <= theta y <= ", max(theta_y))
    print(min(sigma_x), " <= sigma x <= ", max(sigma_x))
    print(min(sigma_y), " <= sigma y <= ", max(sigma_y))
    print(min(tau_xy), " <= tau xy <= ", max(tau_xy))
    # draw_vtk(nodes, elements, u, title="u", show_labels=True)
    # draw_vtk(nodes, elements, v, title="v", show_labels=True)
    # draw_vtk(nodes, elements, w, title="w", show_labels=True)
    # draw_vtk(nodes, elements, theta_x, title="theta x", show_labels=True)
    # draw_vtk(nodes, elements, theta_y, title="theta y", show_labels=True)
    draw_vtk(nodes, elements, sigma_x, title="sigma x", show_labels=True)
    draw_vtk(nodes, elements, sigma_y, title="sigma y", show_labels=True)
    draw_vtk(nodes, elements, tau_xy, title="tau xy", show_labels=True)

    # element_nodes = 4
    # nodes_count = len(nodes)
    # dimension = freedom * nodes_count
    # element_dimension = freedom * element_nodes
    # geometric_matrix = lil_matrix((dimension, dimension))
    # (xi, eta, w) = legendre_quad(3)
    #
    # for element in elements:
    #     local = zeros((element_dimension, element_dimension))
    #     vertices = nodes[element[:], :]
    #     displacement = zeros(freedom * 4)
    #     for j in range(freedom):
    #         displacement[j::freedom] = x[element[:] * freedom + j]
    #
    #     for i in range(len(w)):
    #         (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
    #         bm = array([
    #             [shape_dx[0], 0.0, 0.0, 0.0, 0.0, shape_dx[1], 0.0, 0.0, 0.0, 0.0, shape_dx[2], 0.0, 0.0, 0.0, 0.0,
    #              shape_dx[3], 0.0, 0.0, 0.0, 0.0],
    #             [0.0, shape_dy[0], 0.0, 0.0, 0.0, 0.0, shape_dy[1], 0.0, 0.0, 0.0, 0.0, shape_dy[2], 0.0, 0.0, 0.0, 0.0,
    #              shape_dy[3], 0.0, 0.0, 0.0],
    #             [shape_dy[0], shape_dx[0], 0.0, 0.0, 0.0, shape_dy[1], shape_dx[1], 0.0, 0.0, 0.0, shape_dy[2],
    #              shape_dx[2], 0.0, 0.0, 0.0, shape_dy[3], shape_dx[3], 0.0, 0.0, 0.0]
    #         ])
    #         sigma = d.dot(bm).dot(displacement)
    #         s0 = array([
    #             [sigma[0], sigma[2]],
    #             [sigma[2], sigma[1]]
    #         ])
    #         bm1 = array([
    #             [shape_dx[0], 0.0, 0.0, 0.0, 0.0, shape_dx[1], 0.0, 0.0, 0.0, 0.0, shape_dx[2], 0.0, 0.0, 0.0, 0.0, shape_dx[3], 0.0, 0.0, 0.0, 0.0],
    #             [shape_dy[0], 0.0, 0.0, 0.0, 0.0, shape_dy[1], 0.0, 0.0, 0.0, 0.0, shape_dy[2], 0.0, 0.0, 0.0, 0.0, shape_dy[3], 0.0, 0.0, 0.0, 0.0]
    #         ])
    #         bm2 = array([
    #             [0.0, shape_dx[0], 0.0, 0.0, 0.0, 0.0, shape_dx[1], 0.0, 0.0, 0.0, 0.0, shape_dx[2], 0.0, 0.0, 0.0, 0.0, shape_dx[3], 0.0, 0.0, 0.0],
    #             [0.0, shape_dy[0], 0.0, 0.0, 0.0, 0.0, shape_dy[1], 0.0, 0.0, 0.0, 0.0, shape_dy[2], 0.0, 0.0, 0.0, 0.0, shape_dy[3], 0.0, 0.0, 0.0]
    #         ])
    #
    #     for i in range(element_dimension):
    #         ii = element[i / freedom] * freedom + i % freedom
    #         for j in range(i, element_dimension):
    #             jj = element[j / freedom] * freedom + j % freedom
    #             global_matrix[ii, jj] += local[i, j]
    #             if i != j:
    #                 global_matrix[jj, ii] = global_matrix[ii, jj]