#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import read, draw_vtk
    from assembly2d import assembly_quads_mindlin_plate
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import lil_matrix
    from numpy import array, zeros, hstack
    from quadrature import legendre_quad

    radius = 1.40 # A radius of a plate
    h = 0.06 # A thickness of a plate
    e = 110000. #10920.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    freedom = 3
    element_nodes = 4
    s0 = array([
        [1.964, 0.0],
        [0.0, 1.964]
    ])
    df = plane_stress_isotropic(e, nu)
    dc = array([
        [df[2, 2], 0.0],
        [0.0, df[2, 2]]
    ])
    (nodes, elements) = read('gear.txt')

    stiffness = assembly_quads_mindlin_plate(nodes, elements, h, df, 5)

    nodes_count = len(nodes)
    dimension = freedom * nodes_count
    element_dimension = freedom * element_nodes
    geometric = lil_matrix((dimension, dimension))
    (xi, eta, w) = legendre_quad(5)

    for element in elements:
        kg = zeros((element_dimension, element_dimension))
        vertices = nodes[element[:], :]

        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            bb = array([
                [shape_dx[0], 0.0, 0.0, shape_dx[1], 0.0, 0.0, shape_dx[2], 0.0, 0.0, shape_dx[3], 0.0, 0.0],
                [shape_dy[0], 0.0, 0.0, shape_dy[1], 0.0, 0.0, shape_dy[2], 0.0, 0.0, shape_dy[3], 0.0, 0.0]
            ])
            bs1 = array([
                [0.0, shape_dx[0], 0.0, 0.0, shape_dx[1], 0.0, 0.0, shape_dx[2], 0.0, 0.0, shape_dx[3], 0.0],
                [0.0, shape_dy[0], 0.0, 0.0, shape_dy[1], 0.0, 0.0, shape_dy[2], 0.0, 0.0, shape_dy[3], 0.0]
            ])
            bs2 = array([
                [0.0, 0.0, shape_dx[0], 0.0, 0.0, shape_dx[1], 0.0, 0.0, shape_dx[2], 0.0, 0.0, shape_dx[3]],
                [0.0, 0.0, shape_dy[0], 0.0, 0.0, shape_dy[1], 0.0, 0.0, shape_dy[2], 0.0, 0.0, shape_dy[3]]
            ])
            kg = kg + h * bb.transpose().dot(s0).dot(bb) * jacobian * w[i] + h**3.0 / 12.0 * (bs1.transpose().dot(s0).dot(bs1) + bs2.transpose().dot(s0).dot(bs2)) * jacobian * w[i]

        for i in range(element_dimension):
            ii = element[int(i / freedom)] * freedom + i % freedom
            for j in range(i, element_dimension):
                jj = element[int(j / freedom)] * freedom + j % freedom
                geometric[ii, jj] += kg[i, j]
                if ii != jj:
                    geometric[jj, ii] = geometric[ii, jj]

    print("Assembly is done")
    active = range(dimension)
    for i in range(len(nodes)):
        if abs(nodes[i, 0]**2.0 + nodes[i, 1]**2.0 - radius**2.0) < 0.0000001:
            active.remove(freedom * i)
            active.remove(freedom * i + 1)
            active.remove(freedom * i + 2)

    print ("Boundary conditions are processed")
    geometric = lil_matrix(geometric.tocsr()[:, active])
    geometric = lil_matrix(geometric.tocsc()[active, :])
    geometric = geometric.tocsr()
    stiffness = lil_matrix(stiffness.tocsr()[:, active])
    stiffness = lil_matrix(stiffness.tocsc()[active, :])
    stiffness = stiffness.tocsr()
    vals, vecs = eigsh(A=stiffness, M=geometric, sigma=0.0, which='LM')
    print(vals)
    x = zeros(dimension)
    x[active] = vecs[:, 0]
    w = array(x[0::freedom])
    draw_vtk(nodes=hstack((nodes, w.reshape(len(w), 1)/4.0)), elements=elements, values=w, title="w", show_labels=True, show_axes=True)