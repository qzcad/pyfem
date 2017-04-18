#!/usr/bin/env python
# -*- coding: utf-8 -*-


def assembly_quads_stress_strain(nodes, elements, strain_stress_matrix, gauss_order=2):
    """
    Assembly routine for plane stress-strain state analysis
    :param nodes: Array of nodes coordinates
    :param elements: Array of quads (mesh)
    :param strain_stress_matrix: The stress-strain relations matrix
    :param gauss_order: Order of gaussian quadratures
    :return: Global stiffness matrix in the LIL sparse format (Row-based linked list sparse matrix)
    Order: u_0, v0, u_1, v_1, ..., u_(n-1), v_(n-1), n - nodes count
    """
    from numpy import zeros
    from numpy import array
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from scipy.sparse import lil_matrix
    from print_progress import print_progress
    print "Assembly is started"
    freedom = 2
    element_nodes = 4
    nodes_count = len(nodes)
    dimension = freedom * nodes_count
    element_dimension = freedom * element_nodes
    global_matrix = lil_matrix((dimension, dimension))
    elements_count = len(elements)
    (xi, eta, w) = legendre_quad(gauss_order)
    for element_index in range(elements_count):
        local = zeros((element_dimension, element_dimension))
        element_nodes = nodes[elements[element_index, :], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(element_nodes, xi[i], eta[i])
            b = array([
                [shape_dx[0],   0.0,            shape_dx[1],    0.0,            shape_dx[2],    0.0,            shape_dx[3],    0.0],
                [0.0,           shape_dy[0],    0.0,            shape_dy[1],    0.0,            shape_dy[2],    0.0,            shape_dy[3]],
                [shape_dy[0],   shape_dx[0],    shape_dy[1],    shape_dx[1],    shape_dy[2],    shape_dx[2],    shape_dy[3],    shape_dx[3]]
            ])
            bt = b.conj().transpose()
            local = local + bt.dot(strain_stress_matrix).dot(b) * jacobian * w[i]
        for i in range(element_dimension):
            ii = elements[element_index, i / freedom] * freedom + i % freedom
            for j in range(i, element_dimension):
                jj = elements[element_index, j / freedom] * freedom + j % freedom
                global_matrix[ii, jj] += local[i, j]
                if i != j:
                    global_matrix[jj, ii] = global_matrix[ii, jj]
        print_progress(element_index, elements_count - 1)
    print "\nAssembly is completed"
    return global_matrix


def assembly_triangles_stress_strain(nodes, elements, strain_stress_matrix, gauss_order=2):
    """
    Assembly routine for plane stress-strain state analysis
    :param nodes: Array of nodes coordinates
    :param elements: Array of triangles (mesh)
    :param gauss_order: Order of gaussian quadratures
    :param strain_stress_matrix: The stress-strain relations matrix
    :return: Global stiffness matrix in the LIL sparse format (Row-based linked list sparse matrix)
    Order: u_0, v0, u_1, v_1, ..., u_(n-1), v_(n-1), n - nodes count
    """
    from numpy import zeros
    from numpy import array
    from quadrature import legendre_triangle
    from shape_functions import iso_triangle
    from scipy.sparse import lil_matrix
    from print_progress import print_progress
    print "Assembly is started"
    freedom = 2
    element_nodes = 3
    nodes_count = len(nodes)
    dimension = freedom * nodes_count
    element_dimension = freedom * element_nodes
    global_matrix = lil_matrix((dimension, dimension))
    elements_count = len(elements)
    (xi, eta, w) = legendre_triangle(gauss_order)
    for element_index in range(elements_count):
        local = zeros((element_dimension, element_dimension))
        element_nodes = nodes[elements[element_index, :], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_triangle(element_nodes, xi[i], eta[i])
            b = array([
                [shape_dx[0],   0.0,            shape_dx[1],    0.0,            shape_dx[2],    0.0        ],
                [0.0,           shape_dy[0],    0.0,            shape_dy[1],    0.0,            shape_dy[2]],
                [shape_dy[0],   shape_dx[0],    shape_dy[1],    shape_dx[1],    shape_dy[2],    shape_dx[2]]
            ])
            bt = b.conj().transpose()
            local = local + bt.dot(strain_stress_matrix).dot(b) * jacobian * w[i]
        for i in range(element_dimension):
            ii = elements[element_index, i / freedom] * freedom + i % freedom
            for j in range(i, element_dimension):
                jj = elements[element_index, j / freedom] * freedom + j % freedom
                global_matrix[ii, jj] += local[i, j]
                if i != j:
                    global_matrix[jj, ii] = global_matrix[ii, jj]
        print_progress(element_index, elements_count - 1)
    print "\nAssembly is completed"
    return global_matrix


if __name__ == "__main__":
    from numpy import array
    from mesh2d import rectangular_quads
    from mesh2d import rectangular_triangles
    from plot_coo_matrix import plot_coo_matrix
    d = array([
        [1., 1., 0.],
        [1., 1., 0.],
        [0., 0., 1.]
    ])
    (nodes, elements) = rectangular_quads(x_count=51, y_count=11, x_origin=-10.0, y_origin=-2.0, width=20.0, height=4.0)
    global_matrix = assembly_quads_stress_strain(nodes, elements, d)
    plot_coo_matrix(global_matrix)
    (nodes, elements) = rectangular_triangles(x_count=51, y_count=11, x_origin=-10.0, y_origin=-2.0, width=20.0, height=4.0)
    global_matrix = assembly_triangles_stress_strain(nodes, elements, d)
    plot_coo_matrix(global_matrix)