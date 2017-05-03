#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from mesh2d import rectangular_quads
    from mesh2d import draw_vtk
    from assembly2d import assembly_quads_mindlin_plate
    from assembly2d import assembly_initial_value
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    from force import thermal_force_plate_5
    from scipy.sparse.linalg import spsolve, eigsh, eigs
    from scipy.sparse import lil_matrix, csr_matrix
    from numpy import array, zeros, ix_, hstack
    from quadrature import legendre_quad

    a = 10.0 # A side of a square plate
    factor = 1.25
    h = a / 100.0 # A thickness of a square plate
    e = 1 #10920.0 # The Young's modulus
    nu = 0.3 # The Poisson's ratio
    n = 101
    freedom = 3
    element_nodes = 4
    s0 = array([
        [0.000142857142857, 0.0],
        [0.0, 0.000142857142857]
    ])
    df = plane_stress_isotropic(e, nu)
    dc = array([
        [df[2, 2], 0.0],
        [0.0, df[2, 2]]
    ])
    (nodes, elements) = rectangular_quads(x_count=n, y_count=n, x_origin=0.0, y_origin=0., width=a, height=a/factor)

    stiffness = assembly_quads_mindlin_plate(nodes, elements, h, df, 5)

    nodes_count = len(nodes)
    dimension = freedom * nodes_count
    element_dimension = freedom * element_nodes
    geometric = lil_matrix((dimension, dimension))
    # stiffness = lil_matrix((dimension, dimension))
    (xi, eta, w) = legendre_quad(5)

    for element in elements:
        # local = zeros((element_dimension, element_dimension))
        kg = zeros((element_dimension, element_dimension))
        vertices = nodes[element[:], :]

        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            # bf = array([
            #     [0.0, shape_dx[0], 0.0, 0.0, shape_dx[1], 0.0, 0.0, shape_dx[2], 0.0, 0.0, shape_dx[3], 0.0],
            #     [0.0, 0.0, shape_dy[0], 0.0, 0.0, shape_dy[1], 0.0, 0.0, shape_dy[2], 0.0, 0.0, shape_dy[3]],
            #     [0.0, shape_dy[0], shape_dx[0], 0.0, shape_dy[1], shape_dx[1], 0.0, shape_dy[2], shape_dx[2], 0.0,
            #      shape_dy[3], shape_dx[3]]
            # ])
            # bc = array([
            #     [shape_dx[0], shape[0], 0.0, shape_dx[1], shape[1], 0.0, shape_dx[2], shape[2], 0.0, shape_dx[3],
            #      shape[3], 0.0],
            #     [shape_dy[0], 0.0, shape[0], shape_dy[1], 0.0, shape[1], shape_dy[2], 0.0, shape[2], shape_dy[3], 0.0,
            #      shape[3]]
            # ])
            # local = local + (h ** 3.0 / 12.0 * (bf.transpose().dot(df).dot(bf)) + 5.0/6.0 * h * (bc.transpose().dot(dc).dot(bc))) * jacobian * w[i]
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
            # if i == 0 or i == 3 or i == 6 or i == 9:
            #     ii = element[int(i / 3)]
            # elif i == 1 or i == 4 or i == 7 or i == 10:
            #     ii = element[int(i / 3)] + nodes_count
            # elif i == 2 or i == 5 or i == 8 or i == 11:
            #     ii = element[int(i / 3)] + 2 * nodes_count
            ii = element[int(i / freedom)] * freedom + i % freedom
            for j in range(i, element_dimension):
                # if j == 0 or j == 3 or j == 6 or j == 9:
                #     jj = element[int(j / 3)]
                # elif j == 1 or j == 4 or j == 7 or j == 10:
                #     jj = element[int(j / 3)] + nodes_count
                # elif j == 2 or j == 5 or j == 8 or j == 11:
                #     jj = element[int(j / 3)] + 2 * nodes_count
                jj = element[int(j / freedom)] * freedom + j % freedom
                # stiffness[ii, jj] += local[i, j]
                geometric[ii, jj] += kg[i, j]
                if ii != jj:
                    # stiffness[jj, ii] = stiffness[ii, jj]
                    geometric[jj, ii] = geometric[ii, jj]
    print("Assembly is done")
    active = range(dimension)
    for i in range(len(nodes)):
        if (abs(nodes[i, 0] - 0) < 0.0000001) or (abs(nodes[i, 0] - a) < 0.0000001) or (abs(nodes[i, 1] - 0) < 0.0000001) or (abs(nodes[i, 1] - a/factor) < 0.0000001):
            active.remove(freedom * i)
        # if (abs(nodes[i, 0] - 0) < 0.0000001) or (abs(nodes[i, 0] - a) < 0.0000001):
        #     active.remove(freedom * i)
        #     active.remove(freedom * i + 1)
        #     active.remove(freedom * i + 2)
        # if (abs(nodes[i, 1] - 0) < 0.0000001) or (abs(nodes[i, 1] - a) < 0.0000001):
        #     # active.remove(i)
        #     # active.remove(i + nodes_count)
        #     # active.remove(i + 2 * nodes_count )
        #     active.remove(freedom * i)
        #     active.remove(freedom * i + 1)
        #     active.remove(freedom * i + 2)

    print ("Boundary conditions are processed")
    # print (active)
    geometric = lil_matrix(geometric.tocsr()[:, active])
    geometric = lil_matrix(geometric.tocsc()[active, :])
    geometric = geometric.tocsr()
    stiffness = lil_matrix(stiffness.tocsr()[:, active])
    stiffness = lil_matrix(stiffness.tocsc()[active, :])
    stiffness = stiffness.tocsr()
    print ("Matrices are converted")
    # stiffness = stiffness.tocsr()
    # stiffness2 = csr_matrix(stiffness[ix_(active, active)])
    # geometric_matrix2 = csr_matrix(geometric[ix_(active, active)])
    # print(stiffness)
    vals, vecs = eigsh(A=stiffness, M=geometric, sigma=0.0, which='LM')
    print(vals)
    x = zeros(dimension)
    x[active] = vecs[:, 0]
    w = array(x[0::freedom])
    draw_vtk(nodes=hstack((nodes, w.reshape(len(w), 1) / 100.0)), elements=elements, values=w, title="w", show_labels=True, show_axes=True)