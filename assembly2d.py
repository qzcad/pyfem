#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import zeros
from numpy import array
from scipy.sparse import lil_matrix
from print_progress import print_progress


def assembly_quads_stress_strain(nodes, elements, strain_stress_matrix, gauss_order=2):
    # type: (array, array, array, int) -> lil_matrix
    """
    Assembly Routine for the Plane Stress-Strain State Analysis using a Mesh of Quadrilaterals
    :param nodes: A two-dimensional array of coordinates (nodes)
    :param elements: A two-dimensional array of quads (a mesh)
    :param strain_stress_matrix: A two-dimensional array that represents stress-strain relations
    :param gauss_order: An order of gaussian quadratures (a count of points used to approximate in each direction)
    :return: A global stiffness matrix stored in the LIL sparse format (Row-based linked list sparse matrix)
    Order: u_0, v0, u_1, v_1, ..., u_(n-1), v_(n-1); n is nodes count
    """
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    print "The assembly routine is started."
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
        element = nodes[elements[element_index, :], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(element, xi[i], eta[i])
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
    print "\nThe assembly routine is completed."
    return global_matrix


def assembly_triangles_stress_strain(nodes, elements, strain_stress_matrix, gauss_order=2):
    # type: (array, array, array, int) -> lil_matrix
    """
    Assembly Routine for the Plane Stress-Strain State Analysis using a Mesh of Triangles
    :param nodes: A two-dimensional array of coordinates (nodes)
    :param elements: A two-dimensional array of quads (a mesh)
    :param strain_stress_matrix: A two-dimensional array that represents stress-strain relations
    :param gauss_order: An order of gaussian quadratures (a count of points used to approximate in each direction)
    :return: A global stiffness matrix stored in the LIL sparse format (Row-based linked list sparse matrix)
    Order: u_0, v0, u_1, v_1, ..., u_(n-1), v_(n-1); n is nodes count
    """
    from quadrature import legendre_triangle
    from shape_functions import iso_triangle
    print "The assembly routine is started."
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
        element = nodes[elements[element_index, :], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_triangle(element, xi[i], eta[i])
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
    print "\nThe assembly routine is completed"
    return global_matrix


def assembly_quads_mindlin_plate(nodes, elements, thickness, young, nu, gauss_order=3, kappa=5.0/6.0):
    # type: (array, array, float, float, float, int, float) -> lil_matrix
    """
    Assembly Routine for the Mindlin Plates Analysis
    :param nodes: A two-dimensional array of plate's nodes coordinates
    :param elements: A two-dimensional array of plate's triangles (mesh)
    :param thickness: A thickness of a plate
    :param young: The Young's Modulus of a material
    :param nu: The Poisson's ratio of a material
    :param gauss_order: An order of gaussian quadratures
    :param kappa: The shear correction factor
    :return: Global stiffness matrix in the LIL sparse format (Row-based linked list sparse matrix)
    Order: u_0, v0, u_1, v_1, ..., u_(n-1), v_(n-1); n - nodes count
    """
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from stress_strain_matrix import plane_stress_isotropic
    print "The assembly routine is started"
    freedom = 3
    element_nodes = 4
    nodes_count = len(nodes)
    dimension = freedom * nodes_count
    element_dimension = freedom * element_nodes
    global_matrix = lil_matrix((dimension, dimension))
    elements_count = len(elements)
    (xi, eta, w) = legendre_quad(gauss_order)
    df = plane_stress_isotropic(young, nu)
    dc = array([
        [df[2, 2], 0.0],
        [0.0, df[2, 2]]
    ])
    for element_index in range(elements_count):
        local = zeros((element_dimension, element_dimension))
        element = nodes[elements[element_index, :], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(element, xi[i], eta[i])
            bf = array([
                [0.0, shape_dx[0], 0.0,         0.0, shape_dx[1], 0.0,         0.0, shape_dx[2], 0.0,         0.0, shape_dx[3], 0.0],
                [0.0, 0.0,         shape_dy[0], 0.0, 0.0,         shape_dy[1], 0.0, 0.0,         shape_dy[2], 0.0, 0.0,         shape_dy[3]],
                [0.0, shape_dy[0], shape_dx[0], 0.0, shape_dy[1], shape_dx[1], 0.0, shape_dy[2], shape_dx[2], 0.0, shape_dy[3], shape_dx[3]]
            ])
            bc = array([
                [shape_dx[0], shape[0], 0.0, shape_dx[1], shape[1], 0.0, shape_dx[2], shape[2], 0.0, shape_dx[3], shape[3], 0.0],
                [shape_dy[0], 0.0, shape[0], shape_dy[1], 0.0, shape[1], shape_dy[2], 0.0, shape[2], shape_dy[3], 0.0, shape[3]]
            ])
            local = local + (thickness**3.0 / 12.0 * (bf.transpose().dot(df).dot(bf)) + kappa * thickness * (bc.transpose().dot(dc).dot(bc))) * jacobian * w[i]
        for i in range(element_dimension):
            ii = elements[element_index, i / freedom] * freedom + i % freedom
            for j in range(i, element_dimension):
                jj = elements[element_index, j / freedom] * freedom + j % freedom
                global_matrix[ii, jj] += local[i, j]
                if i != j:
                    global_matrix[jj, ii] = global_matrix[ii, jj]
        print_progress(element_index, elements_count - 1)
    print "\nThe assembly routine is completed"
    return global_matrix

if __name__ == "__main__":
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
    (nodes, elements) = rectangular_quads(x_count=31, y_count=31, x_origin=0.0, y_origin=0, width=1, height=1)
    global_matrix = assembly_quads_mindlin_plate(nodes, elements, 0.1, 10920, 0.3)
    plot_coo_matrix(global_matrix)