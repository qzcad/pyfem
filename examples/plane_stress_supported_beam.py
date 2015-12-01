#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_stress_strain
    from numpy import array
    from numpy import zeros
    from scipy.sparse.linalg import cg
    l = 10.0  # beam half-length
    c = 2.0  # beam half-height
    e = 203200.0  # Young's modulus
    nu = 0.27  # Poison's modulus
    q = 200.0  # uniformly distributed load
    n = 51  # nodes in the first direction of computational domain
    m = 11  # nodes in the second direction of computational domain
    gauss_order = 3
    d = array([
        [1.0, nu / (1.0 - nu), 0.0],
        [nu / (1.0 - nu), 1.0, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))]
    ]) * e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    print(d)
    (nodes, elements) = rectangular_quads(x_count=n, y_count=m, x_origin=-l, y_origin=-c, width=2.0 * l, height=2.0 * c)
    stiffness = assembly_quads_stress_strain(nodes, elements, d)
    dimension = stiffness.shape[0]
    force = zeros(dimension)
    for i in range(len(nodes)):
        if abs(c - nodes[i, 1]) < 0.0000001:
            force[2 * i + 1] = -(q * 2.0 * l / (n - 1))
            if (abs(l - nodes[i, 0]) < 0.0000001) or (abs(-l - nodes[i, 0]) < 0.0000001):
                force[2 * i + 1] = -(q * 2.0 * l / (n - 1)) / 2.0
    for i in range(len(nodes)):
        if abs(-c - nodes[i, 1]) < 0.0000001 and (abs(l - nodes[i, 0]) < 0.0000001 or abs(-l - nodes[i, 0]) < 0.0000001):
            for j in range(dimension):
                if stiffness[2 * i + 1, j] != 0.0:
                    stiffness[2 * i + 1, j] = 0.0
                if stiffness[j, 2 * i + 1] != 0.0:
                    stiffness[j, 2 * i + 1] = 0.0
            stiffness[2 * i + 1, 2 * i + 1] = 1.0
            force[2 * i + 1] = 0.0
    stiffness = stiffness.tocsr()
    x, info = cg(stiffness, force, tol=1e-8)
    draw_vtk(nodes, elements, x[0::2], title="U")
    draw_vtk(nodes, elements, x[1::2], title="V")


