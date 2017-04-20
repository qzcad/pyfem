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


def interval(func, a, b, count):
    """
    This routine integrates a function over an interval [a; b] using Gauss-Legendre rules
    :param a: Lower limit of the interval
    :param b: Upper limit of the interval
    :param func: A Python function or method to integrate
    :param count: A count of quadrature points
    :return:
    """
    from quadrature import legendre_interval
    from scipy.spatial.distance import euclidean
    (p, w) = legendre_interval(count)
    j = euclidean(a, b) / 2.0
    s0 = 0.5 * (1.0 - p[0]) * func(0.5 * a * (1.0 - p[0]) + 0.5 * b * (1.0 + p[0])) * w[0] * j
    s1 = 0.5 * (1.0 + p[0]) * func(0.5 * a * (1.0 - p[0]) + 0.5 * b * (1.0 + p[0])) * w[0] * j
    for i in range(1, count):
        s0 += 0.5 * (1.0 - p[i]) * func(0.5 * a * (1.0 - p[i]) + 0.5 * b * (1.0 + p[i])) * w[i] * j
        s1 += 0.5 * (1.0 + p[i]) * func(0.5 * a * (1.0 - p[i]) + 0.5 * b * (1.0 + p[i])) * w[i] * j
    return [s0, s1]


def edge_force_quads(nodes, elements, freedom, force_function, gauss_order):
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
        for j in range(len(f0)):
            force[freedom * quad[0] + j] += f0[j]
            force[freedom * quad[1] + j] += f1[j]

        [f0, f1] = interval(force_function, b, c, gauss_order)
        for j in range(len(f0)):
            force[freedom * quad[1] + j] += f0[j]
            force[freedom * quad[2] + j] += f1[j]

        [f0, f1] = interval(force_function, c, d, gauss_order)
        for j in range(len(f0)):
            force[freedom * quad[2] + j] += f0[j]
            force[freedom * quad[3] + j] += f1[j]

        [f0, f1] = interval(force_function, d, a, gauss_order)
        for j in range(len(f0)):
            force[freedom * quad[3] + j] += f0[j]
            force[freedom * quad[0] + j] += f1[j]

    return force


def edge_force_triangles(nodes, elements, freedom, force_function, gauss_order):
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
        for j in range(len(f0)):
            force[freedom * tri[0] + j] += f0[j]
            force[freedom * tri[1] + j] += f1[j]

        [f0, f1] = interval(force_function, b, c, gauss_order)
        for j in range(len(f0)):
            force[freedom * tri[1] + j] += f0[j]
            force[freedom * tri[2] + j] += f1[j]

        [f0, f1] = interval(force_function, c, a, gauss_order)
        for j in range(len(f0)):
            force[freedom * tri[2] + j] += f0[j]
            force[freedom * tri[0] + j] += f1[j]

    return force
