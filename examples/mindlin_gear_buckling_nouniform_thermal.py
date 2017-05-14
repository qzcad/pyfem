#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import draw_vtk, read
    from assembly2d import assembly_quads_stress_strain, assembly_initial_value, assembly_quads_mindlin_plate, assembly_quads_mindlin_plate_geometric
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    from force import thermal_force_quads
    from scipy.sparse.linalg import cg, eigsh
    from scipy.sparse import lil_matrix
    from numpy import array, zeros, hstack
    from math import sqrt

    radius = 1.40  # A radius of a plate
    h = 0.06 # A thickness of a plate
    e = 110.0 # The Young's modulus, MPa
    nu = 0.3 # The Poisson's ratio
    alpha = 1.25E-5
    freedom = 2
    df = plane_stress_isotropic(e, nu)

    (nodes, elements) = read("gear.txt")

    stiffness = assembly_quads_stress_strain(nodes=nodes, elements=elements, elasticity_matrix=df, thickness=h)

    print("Evaluating force...")


    def tfunc(x, y):
        r = sqrt(x**2.0 + y**2.0)
        return 250.0 * (r - 0.08) / (0.13 - 0.08)


    force = thermal_force_quads(nodes=nodes, elements=elements, thickness=h, elasticity_matrix=df, alpha_t=alpha, tfunc=tfunc)

    print("Evaluating boundary conditions...")
    for i in range(len(nodes)):
        if abs(nodes[i, 0]) < 0.0000001:
            assembly_initial_value(stiffness, force, freedom * i, 0.0)
        if abs(nodes[i, 1]) < 0.0000001:
            assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)

    print("Solving a system of linear equations")

    x, info = cg(stiffness, force, tol=1e-10, maxiter=2000)

    u = x[0::freedom]
    v = x[1::freedom]

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
            sigma = df.dot(b).dot(displacement)
            sigma_x[element[i]] += sigma[0]
            sigma_y[element[i]] += sigma[1]
            tau_xy[element[i]] += sigma[2]
            adjacent[element[i]] += 1.0

    sigma_x /= adjacent
    sigma_y /= adjacent
    tau_xy /= adjacent

    print(min(u), " <= u <= ", max(u))
    print(min(v), " <= v <= ", max(v))
    print(min(sigma_x), " <= sigma x <= ", max(sigma_x))
    print(min(sigma_y), " <= sigma y <= ", max(sigma_y))
    print(min(tau_xy), " <= tau xy <= ", max(tau_xy))
    # draw_vtk(nodes, elements, u, title="u", show_labels=True)
    # draw_vtk(nodes, elements, v, title="v", show_labels=True)
    # draw_vtk(nodes, elements, sigma_x, title="sigma x", show_labels=True)
    # draw_vtk(nodes, elements, sigma_y, title="sigma y", show_labels=True)
    # draw_vtk(nodes, elements, tau_xy, title="tau xy", show_labels=True)

    stiffness = assembly_quads_mindlin_plate(nodes, elements, h, df, 5)

    geometric = assembly_quads_mindlin_plate_geometric(nodes=nodes, elements=elements, thickness=h, sigma_x=sigma_x, sigma_y=sigma_y, tau_xy=tau_xy)

    freedom = 3
    plate_dim = freedom * len(nodes)
    active = range(plate_dim)
    for i in range(len(nodes)):
        if abs(nodes[i, 0] ** 2.0 + nodes[i, 1] ** 2.0 - radius ** 2.0) < 0.0000001:
            active.remove(freedom * i)
            # active.remove(freedom * i + 1)
            # active.remove(freedom * i + 2)

    print ("Boundary conditions are processed")
    geometric = lil_matrix(geometric.tocsr()[:, active])
    geometric = lil_matrix(geometric.tocsc()[active, :])
    geometric = geometric.tocsr()
    stiffness = lil_matrix(stiffness.tocsr()[:, active])
    stiffness = lil_matrix(stiffness.tocsc()[active, :])
    stiffness = stiffness.tocsr()
    vals, vecs = eigsh(A=stiffness, M=geometric, sigma=0.0, which='LM')
    print(vals)
    print (vals[0] * 250.0 * (radius - 0.08) / (0.13 - 0.08))
    for i in range(len(vals)):
        x = zeros(plate_dim)
        x[active] = vecs[:, i]
        w = array(x[0::freedom])
        draw_vtk(nodes=hstack((nodes, w.reshape(len(w), 1) / 1.0)), elements=elements, values=w, title="w, T = " + str(vals[i] * 250.0 * (radius - 0.08) / (0.13 - 0.08)),
                 show_labels=True, show_axes=True)
