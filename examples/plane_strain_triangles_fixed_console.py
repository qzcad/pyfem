#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_triangles
    from mesh2d import draw_vtk
    from assembly2d import assembly_triangles_stress_strain
    from stress_strain_matrix import plane_strain_isotropic
    from numpy import zeros
    from scipy.sparse.linalg import spsolve
    l = 10.0  # beam half-length
    c = 2.0  # beam half-height
    e = 203200.0  # Young's modulus
    nu = 0.27  # Poison's modulus
    q = 100.0  # uniformly distributed load
    n = 51  # nodes in the first direction of computational domain
    m = 21  # nodes in the second direction of computational domain
    d = plane_strain_isotropic(e, nu)
    print(d)
    (nodes, elements) = rectangular_triangles(x_count=n, y_count=m, x_origin=0.0, y_origin=-c, width=l, height=2.0 * c)
    stiffness = assembly_triangles_stress_strain(nodes, elements, d)
    dimension = stiffness.shape[0]
    force = zeros(dimension)
    for i in range(len(nodes)):
        if abs(nodes[i, 0]) < 0.0000001:
            force[2 * i + 1] = -q / m
    for i in range(len(nodes)):
        if abs(l - nodes[i, 0]) < 0.0000001:
            for j in range(dimension):
                if stiffness[2 * i, j] != 0.0:
                    stiffness[2 * i, j] = 0.0
                if stiffness[j, 2 * i] != 0.0:
                    stiffness[j, 2 * i] = 0.0
                if stiffness[2 * i + 1, j] != 0.0:
                    stiffness[2 * i + 1, j] = 0.0
                if stiffness[j, 2 * i + 1] != 0.0:
                    stiffness[j, 2 * i + 1] = 0.0
            stiffness[2 * i, 2 * i] = 1.0
            force[2 * i] = 0.0
            stiffness[2 * i + 1, 2 * i + 1] = 1.0
            force[2 * i + 1] = 0.0
    stiffness = stiffness.tocsr()
    x = spsolve(stiffness, force)
    print(min(x[0::2]), " <= X <= ", max(x[0::2]))
    print(min(x[1::2]), " <= Y <= ", max(x[1::2]))
    draw_vtk(nodes, elements, x[0::2], title="X", show_labels=True)
    draw_vtk(nodes, elements, x[1::2], title="Y", show_labels=True)
