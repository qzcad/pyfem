#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_mindlin_plate
    from numpy import zeros
    from quadrature import legendre_quad
    from  shape_functions import iso_quad
    from scipy.sparse.linalg import spsolve
    from numpy import array
    from force import volume_force_quads
    a = 1.0 # A side of a square plate
    h = 0.1 # A thickness of a square plate
    e = 10920.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    q = 1.0 # A load intensity
    n = 31
    (nodes, elements) = rectangular_quads(x_count=n, y_count=n, x_origin=0.0, y_origin=0., width=a, height=a)
    stiffness = assembly_quads_mindlin_plate(nodes, elements, h, e, nu)
    # force
    dimension = stiffness.shape[0]

    def force_func(node): return array([q, 0.0, 0.0])

    force = volume_force_quads(nodes=nodes, elements=elements, thickness=1.0, freedom=3, force_function=force_func, gauss_order=3)

    # boundary conditions
    for i in range(len(nodes)):
        if (abs(nodes[i, 0] - 0) < 0.0000001) or (abs(nodes[i, 0] - a) < 0.0000001) or (abs(nodes[i, 1] - 0) < 0.0000001) or (abs(nodes[i, 1] - a) < 0.0000001):
            for j in range(dimension):
                if stiffness[3 * i, j] != 0.0:
                    stiffness[3 * i, j] = 0.0
                if stiffness[j, 3 * i] != 0.0:
                    stiffness[j, 3 * i] = 0.0
            stiffness[3 * i, 3 * i] = 1.0
            force[3 * i] = 0.0
    # linear system
    stiffness = stiffness.tocsr()
    x = spsolve(stiffness, force)
    w = x[0::3]
    theta_x = x[1::3]
    theta_y = x[2::3]
    print(min(w), " <= w <= ", max(w))
    print(min(theta_x), " <= theta x <= ", max(theta_x))
    print(min(theta_y), " <= theta y <= ", max(theta_y))
    draw_vtk(nodes, elements, w, title="w", show_labels=True)
    draw_vtk(nodes, elements, theta_x, title="theta x", show_labels=True)
    draw_vtk(nodes, elements, theta_y, title="theta y", show_labels=True)