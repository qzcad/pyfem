#!/usr/bin/env python
# -*- coding: utf-8 -*-


def assembly_quads_stress_strain(nodes, elements, strain_stress_matrix):
    from numpy import zeros
    from numpy import array
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from scipy.sparse import lil_matrix
    freedom = 2
    element_nodes = 4
    dimension = freedom * len(nodes)
    element_dimension = freedom * element_nodes
    global_matrix = lil_matrix((dimension, dimension))
    (xi, eta, w) = legendre_quad(2)
    for element_number in range(len(elements)):
        local = zeros((element_dimension, element_dimension))
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(nodes, elements, element_number, xi[i], eta[i])
            b = array([
                [shape_dx[0],   0.0,            shape_dx[1],    0.0,            shape_dx[2],    0.0,            shape_dx[3],    0.0],
                [0.0,           shape_dy[0],    0.0,            shape_dy[1],    0.0,            shape_dy[2],    0.0,            shape_dy[3]],
                [shape_dy[0],   shape_dx[0],    shape_dy[1],    shape_dx[1],    shape_dy[2],    shape_dx[2],    shape_dy[3],    shape_dx[3]]
            ])
            bt = b.conj().transpose()
            local = local + bt.dot(strain_stress_matrix).dot(b) * jacobian * w[i]
        # todo: assembly routine
