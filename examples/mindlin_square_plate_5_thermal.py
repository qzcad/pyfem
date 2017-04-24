#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_mindlin_plate_laminated
    from assembly2d import assembly_initial_value
    from stress_strain_matrix import plane_stress_isotropic
    from force import thermal_force_plate_5
    from scipy.sparse.linalg import spsolve
    from numpy import array

    a = 10.0 # A side of a square plate
    h = a / 100.0 # A thickness of a square plate
    e = 1.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    alpha = 1.0E-6
    n = 101
    freedom = 5
    (nodes, elements) = rectangular_quads(x_count=n, y_count=n, x_origin=0.0, y_origin=0., width=a, height=a)

    stiffness = assembly_quads_mindlin_plate_laminated(nodes, elements, [h], [plane_stress_isotropic(e, nu)])

    print("Evaluating force...")

    def force_func(node): return array([0.0, 0.0, q, 0.0, 0.0])

    force = thermal_force_plate_5(nodes=nodes, elements=elements, thicknesses=[h], elasticity_matrices=[plane_stress_isotropic(e, nu)], alpha_t=alpha)

    print("Evaluating boundary conditions...")
    for i in range(len(nodes)):
        # if (abs(nodes[i, 0] - 0) < 0.0000001) or (abs(nodes[i, 0] - a) < 0.0000001):
        #     assembly_initial_value(stiffness, force, freedom * i, 0.0)
        #     # assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)
        #     assembly_initial_value(stiffness, force, freedom * i + 2, 0.0)
        #     # assembly_initial_value(stiffness, force, freedom * i + 3, 0.0)
        #     assembly_initial_value(stiffness, force, freedom * i + 4, 0.0)
        if (abs(nodes[i, 1] - 0) < 0.0000001) or (abs(nodes[i, 1] - a) < 0.0000001):
            assembly_initial_value(stiffness, force, freedom * i, 0.0)
            assembly_initial_value(stiffness, force, freedom * i + 1, 0.0)
            assembly_initial_value(stiffness, force, freedom * i + 2, 0.0)
            assembly_initial_value(stiffness, force, freedom * i + 3, 0.0)
            assembly_initial_value(stiffness, force, freedom * i + 4, 0.0)

    print("Solving a system of linear equations")

    x = spsolve(stiffness, force)

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
    draw_vtk(nodes, elements, u, title="u", show_labels=True)
    draw_vtk(nodes, elements, v, title="v", show_labels=True)
    draw_vtk(nodes, elements, w, title="w", show_labels=True)
    draw_vtk(nodes, elements, theta_x, title="theta x", show_labels=True)
    draw_vtk(nodes, elements, theta_y, title="theta y", show_labels=True)
