#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_stress_strain
    from assembly2d import assembly_initial_value
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    from force import thermal_force_quads
    from scipy.sparse.linalg import spsolve, eigsh
    from scipy.sparse import lil_matrix, csr_matrix
    from numpy import array, zeros, ix_
    from quadrature import legendre_quad

    a = 10.0 # A side of a square plate
    factor = 3.0
    b = a / factor
    h = a / 100.0 # A thickness of a square plate
    e = 1.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    alpha = 1.0E-4
    n = 101
    freedom = 3
    d = plane_stress_isotropic(e, nu)
    (nodes, elements) = rectangular_quads(x_count=n, y_count=int(n/factor), x_origin=0.0, y_origin=0., width=a, height=b)

    stiffness = assembly_quads_stress_strain(nodes=nodes, elements=elements, elasticity_matrix=d, thickness=h)

    print("Evaluating force...")

    force = thermal_force_quads(nodes=nodes, elements=elements, thickness=h, elasticity_matrices=d, alpha_t=alpha)

    print("Evaluating boundary conditions...")
    for i in range(len(nodes)):
        if abs(nodes[i, 0] - a/2.0) < 0.0000001:
            assembly_initial_value(stiffness, force, freedom * i, 0.0)
        if abs(nodes[i, 1] - b/2.0) < 0.0000001:
            assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)

    print("Solving a system of linear equations")

    x = spsolve(stiffness, force)

    xi = array([-1.0, 1.0, 1.0, -1.0])
    eta = array([-1.0, -1.0, 1.0, 1.0])
    nodes_count = len(nodes)
    sigma_x = zeros(nodes_count)
    sigma_y = zeros(nodes_count)
    tau_xy = zeros(nodes_count)
    adjacent = zeros(nodes_count)
    for element in elements:
        vertices = nodes[element[:], :]
        displacement = zeros(freedom * 4)
        for j in range(freedom):
            displacement[j::freedom] = x[element[:] * freedom + j]

        for i in range(4):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            b = array([
                [shape_dx[0], 0.0, shape_dx[1], 0.0, shape_dx[2], 0.0, shape_dx[3], 0.0],
                [0.0, shape_dy[0], 0.0, shape_dy[1], 0.0, shape_dy[2], 0.0, shape_dy[3]],
                [shape_dy[0], shape_dx[0], shape_dy[1], shape_dx[1], shape_dy[2], shape_dx[2], shape_dy[3], shape_dx[3]]
            ])
            sigma = d.dot(b).dot(displacement)
            sigma_x[element[i]] += sigma[0]
            sigma_y[element[i]] += sigma[1]
            tau_xy[element[i]] += sigma[2]
            adjacent[element[i]] += 1.0

    sigma_x /= adjacent
    sigma_y /= adjacent
    tau_xy /= adjacent
    u = x[0::freedom]
    v = x[1::freedom]
    print(min(u), " <= u <= ", max(u))
    print(min(v), " <= v <= ", max(v))
    print(min(sigma_x), " <= sigma x <= ", max(sigma_x))
    print(min(sigma_y), " <= sigma y <= ", max(sigma_y))
    print(min(tau_xy), " <= tau xy <= ", max(tau_xy))
    print("Analytical sigma: " + str(e * alpha / (1 - nu)))
    # draw_vtk(nodes, elements, u, title="u", show_labels=True)
    # draw_vtk(nodes, elements, v, title="v", show_labels=True)
    # draw_vtk(nodes, elements, w, title="w", show_labels=True)
    # draw_vtk(nodes, elements, theta_x, title="theta x", show_labels=True)
    # draw_vtk(nodes, elements, theta_y, title="theta y", show_labels=True)
    draw_vtk(nodes, elements, sigma_x, title="sigma x", show_labels=True)
    draw_vtk(nodes, elements, sigma_y, title="sigma y", show_labels=True)
    # draw_vtk(nodes, elements, tau_xy, title="tau xy", show_labels=True)
