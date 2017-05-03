#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import zeros


def nodal_force(nodes, freedom, force_function):
    """
    Assembly routine for nodal forces processing
    :param nodes: A two dimensional array of coordinates
    :param freedom: A count of freedoms in each node
    :param force_function: The function that returns a value of a force vector 
    :return: An array of values
    """
    nodes_count = len(nodes)
    force = zeros(freedom * nodes_count)
    for i in range(nodes_count):
        f = force_function(nodes[i, :])
        for j in range(len(f)):
            force[freedom * i + j] += f[j]
    return force


def interval(func, a, b, gauss_order):
    """
    This routine integrates a function over an interval [a; b] using Gauss-Legendre rules
    :param a: Lower limit of the interval
    :param b: Upper limit of the interval
    :param func: A Python function or method to integrate
    :param gauss_order: A count of quadrature points
    :return:
    """
    from quadrature import legendre_interval
    from scipy.spatial.distance import euclidean
    (p, w) = legendre_interval(gauss_order)
    j = euclidean(a, b) / 2.0
    s0 = 0.5 * (1.0 - p[0]) * func(0.5 * a * (1.0 - p[0]) + 0.5 * b * (1.0 + p[0])) * w[0] * j
    s1 = 0.5 * (1.0 + p[0]) * func(0.5 * a * (1.0 - p[0]) + 0.5 * b * (1.0 + p[0])) * w[0] * j
    for i in range(1, gauss_order):
        s0 += 0.5 * (1.0 - p[i]) * func(0.5 * a * (1.0 - p[i]) + 0.5 * b * (1.0 + p[i])) * w[i] * j
        s1 += 0.5 * (1.0 + p[i]) * func(0.5 * a * (1.0 - p[i]) + 0.5 * b * (1.0 + p[i])) * w[i] * j
    return [s0, s1]


def edge_force_quads(nodes, elements, freedom, force_function, gauss_order=3):
    """
    Assembly routine for processing of forces distributed over edges (quadrilaterals)
    :param nodes: A two dimensional array of coordinates
    :param elements: A two dimensional array of elements (cells)
    :param freedom: A count of freedom in each node
    :param force_function: The function that returns a value of a force
    :param gauss_order: A count of Gauss-Legendre quadratures point used for integration
    :return: An array of values
    """
    nodes_count = len(nodes)
    force = zeros(freedom * nodes_count)
    for quad in elements:
        a = nodes[quad[0]]
        b = nodes[quad[1]]
        c = nodes[quad[2]]
        d = nodes[quad[3]]
        [f0, f1] = interval(force_function, a, b, gauss_order)
        [f2, f3] = interval(force_function, b, c, gauss_order)
        [f4, f5] = interval(force_function, c, d, gauss_order)
        [f6, f7] = interval(force_function, d, a, gauss_order)
        for j in range(len(f0)):
            force[freedom * quad[0] + j] += f0[j]
            force[freedom * quad[1] + j] += f1[j]
            force[freedom * quad[1] + j] += f2[j]
            force[freedom * quad[2] + j] += f3[j]
            force[freedom * quad[2] + j] += f4[j]
            force[freedom * quad[3] + j] += f5[j]
            force[freedom * quad[3] + j] += f6[j]
            force[freedom * quad[0] + j] += f7[j]

    return force


def edge_force_triangles(nodes, elements, freedom, force_function, gauss_order=3):
    """
    Assembly routine for processing of forces distributed over edges (triangles)
    :param nodes: A two dimensional array of coordinates
    :param elements: A two dimensional array of elements (cells)
    :param freedom: A count of freedom in each node
    :param force_function: The function that returns a value of a force
    :param gauss_order: A count of Gauss-Legendre quadratures point used for integration
    :return: An array of values
    """
    nodes_count = len(nodes)
    force = zeros(freedom * nodes_count)
    for tri in elements:
        a = nodes[tri[0]]
        b = nodes[tri[1]]
        c = nodes[tri[2]]
        [f0, f1] = interval(force_function, a, b, gauss_order)
        [f2, f3] = interval(force_function, b, c, gauss_order)
        [f4, f5] = interval(force_function, c, a, gauss_order)
        for j in range(freedom):
            force[freedom * tri[0] + j] += f0[j]
            force[freedom * tri[1] + j] += f1[j]
            force[freedom * tri[1] + j] += f2[j]
            force[freedom * tri[2] + j] += f3[j]
            force[freedom * tri[2] + j] += f4[j]
            force[freedom * tri[0] + j] += f5[j]

    return force


def volume_force_quads(nodes, elements, thickness, freedom, force_function, gauss_order=3):
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from numpy import sum
    dimension = len(nodes) * freedom
    force = zeros(dimension)
    element_nodes = 4
    (xi, eta, w) = legendre_quad(gauss_order)
    directions_number = len(nodes[0])
    for element in elements:
        fe = zeros([element_nodes, freedom])
        vertices = nodes[element[:], :]
        node = zeros([element_nodes, directions_number])
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            for j in range(directions_number):
                node[:, j] = sum(shape * vertices[:, j])
                node[:, j] = sum(shape * vertices[:, j])
            for j in range(element_nodes):
                fe[j, :] = fe[j, :] + shape[j] * force_function(node) * thickness * w[i] * jacobian
        for i in range(element_nodes):
            for j in range(freedom):
                force[element[i] * freedom + j] += fe[i, j]

    return force


def thermal_force_quads(nodes, elements, thickness, elasticity_matrix, alpha_t, gauss_order=3):
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from numpy import sum, array
    freedom = 2
    dimension = len(nodes) * freedom
    force = zeros(dimension)
    element_nodes = 4
    (xi, eta, w) = legendre_quad(gauss_order)
    alpha = array([alpha_t, alpha_t, 0.0])
    for element in elements:
        fe = zeros(element_nodes * freedom)
        vertices = nodes[element[:], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            b = array([
                [shape_dx[0], 0.0, shape_dx[1], 0.0, shape_dx[2], 0.0, shape_dx[3], 0.0],
                [0.0, shape_dy[0], 0.0, shape_dy[1], 0.0, shape_dy[2], 0.0, shape_dy[3]],
                [shape_dy[0], shape_dx[0], shape_dy[1], shape_dx[1], shape_dy[2], shape_dx[2], shape_dy[3], shape_dx[3]]
            ])
            fe = fe + thickness * b.transpose().dot(elasticity_matrix).dot(alpha) * w[i] * jacobian

        for i in range(element_nodes * freedom):
            ii = element[i / freedom] * freedom + i % freedom
            force[ii] += fe[i]

    return force


def thermal_force_plate_5(nodes, elements, thicknesses, elasticity_matrices, alpha_t, gauss_order=3):
    from quadrature import legendre_quad
    from shape_functions import iso_quad
    from numpy import sum, array
    freedom = 5
    dimension = len(nodes) * freedom
    force = zeros(dimension)
    element_nodes = 4
    (xi, eta, w) = legendre_quad(gauss_order)
    alpha = array([alpha_t, alpha_t, 0.0])
    h = sum(thicknesses)
    for element in elements:
        fe = zeros(element_nodes * freedom)
        vertices = nodes[element[:], :]
        for i in range(len(w)):
            (jacobian, shape, shape_dx, shape_dy) = iso_quad(vertices, xi[i], eta[i])
            bm = array([
                [shape_dx[0], 0.0, 0.0, 0.0, 0.0, shape_dx[1], 0.0, 0.0, 0.0, 0.0, shape_dx[2], 0.0, 0.0, 0.0, 0.0,
                 shape_dx[3], 0.0, 0.0, 0.0, 0.0],
                [0.0, shape_dy[0], 0.0, 0.0, 0.0, 0.0, shape_dy[1], 0.0, 0.0, 0.0, 0.0, shape_dy[2], 0.0, 0.0, 0.0, 0.0,
                 shape_dy[3], 0.0, 0.0, 0.0],
                [shape_dy[0], shape_dx[0], 0.0, 0.0, 0.0, shape_dy[1], shape_dx[1], 0.0, 0.0, 0.0, shape_dy[2],
                 shape_dx[2], 0.0, 0.0, 0.0, shape_dy[3], shape_dx[3], 0.0, 0.0, 0.0]
            ])
            z0 = -h / 2.0
            for j in range(len(thicknesses)):
                z1 = z0 + thicknesses[j]
                df = elasticity_matrices[j]
                fe = fe + (z1 - z0) * bm.transpose().dot(df).dot(alpha) * w[i] * jacobian
                z0 = z1

        for i in range(element_nodes * freedom):
            ii = element[i / freedom] * freedom + i % freedom
            force[ii] += fe[i]

    return force
