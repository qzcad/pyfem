#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_stress_strain
    from assembly2d import assembly_initial_value
    from force import nodal_force
    from stress_strain_matrix import plane_strain_isotropic
    from scipy.sparse.linalg import spsolve
    from numpy import array

    l = 10.0  # beam half-length
    c = 2.0  # beam half-height
    e = 203200.0  # Young's modulus
    nu = 0.27  # Poison's modulus
    q = 100.0  # uniformly distributed load
    n = 51  # nodes in the first direction of computational domain
    m = 21  # nodes in the second direction of computational domain

    d = plane_strain_isotropic(e, nu)
    print(d)

    (nodes, elements) = rectangular_quads(x_count=n, y_count=m, x_origin=0.0, y_origin=-c, width=l, height=2.0 * c)
    stiffness = assembly_quads_stress_strain(nodes=nodes, elements=elements, thickness=1.0, elasticity_matrix=d)

    print("Evaluating force...")


    def force_func(node):
        if abs(node[0]) < 0.0000001:
            return array([0.0, -q / m])
        else:
            return array([0.0, 0.0])


    force = nodal_force(nodes=nodes, freedom=2, force_function=force_func)

    print("Evaluating boundary conditions...")
    for i in range(len(nodes)):
        if abs(l - nodes[i, 0]) < 0.0000001:
            assembly_initial_value(stiffness, force, 2 * i, 0.0)
            assembly_initial_value(stiffness, force, 2 * i + 1, 0.0)

    print("Solving a system of linear equations")
    x = spsolve(stiffness, force)

    u = x[0::2]
    v = x[1::2]
    print(min(u), " <= u <= ", max(u))
    print(min(v), " <= Y <= ", max(v))
    draw_vtk(nodes, elements, u, title="u", show_labels=True)
    draw_vtk(nodes, elements, v, title="v", show_labels=True)
