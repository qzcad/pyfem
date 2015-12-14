#!/usr/bin/env python
# -*- coding: utf-8 -*-


def iso_quad(nodes, elements, element_index, xi, eta):
    """
    Isoparametric shape function for a quadrilateral element (nodes must be ordered counterclockwise)
    :param nodes: [nodes_count; 2]-matrix
    :param elements: [elements_count; 4]-matrix
    :param element_index: Index of element is being approximated
    :param xi: Coordinate in the first parametric direction
    :param eta: Coordinate in the second parametric direction
    :return: Tuple: jacobian, array of the shape functions, array of the shape functions derivatives in the first
    direction of computational domain, array of the shape functions derivatives in the second direction of computational
    domain
    """
    from numpy import array
    from numpy import sum
    from numpy.linalg import det
    from numpy.linalg import inv
    shape = array([
        (1.0 - xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 + eta) / 4.0,
        (1.0 - xi) * (1.0 + eta) / 4.0
    ])  # bilinear shape functions
    shape_dxi = array([
        -(1.0 - eta) / 4.0,
        (1.0 - eta) / 4.0,
        (1.0 + eta) / 4.0,
        -(1.0 + eta) / 4.0
    ])  # derivatives of the shape functions in the first parametric direction
    shape_deta = array([
        -(1.0 - xi) / 4.0,
        -(1.0 + xi) / 4.0,
        (1.0 + xi) / 4.0,
        (1.0 - xi) / 4.0
    ])  # derivatives of the shape functions in the second parametric direction
    x = nodes[elements[element_index, :], 0]
    y = nodes[elements[element_index, :], 1]
    jacobi = array([
        [sum(shape_dxi * x), sum(shape_dxi * y)],
        [sum(shape_deta * x), sum(shape_deta * y)]
    ])  # Jacobi matrix
    jacobian = det(jacobi)
    inverted_jacobi = inv(jacobi)
    shape_dx = inverted_jacobi[0, 0] * shape_dxi + inverted_jacobi[0, 1] * shape_deta
    shape_dy = inverted_jacobi[1, 0] * shape_dxi + inverted_jacobi[1, 1] * shape_deta
    return jacobian, shape, shape_dx, shape_dy


def iso_triangle(nodes, elements, element_index, xi, eta):
    """
    Isoparametric shape function for a triangular element (nodes must be ordered counterclockwise)
    :param nodes: [nodes_count; 2]-matrix
    :param elements: [elements_count; 3]-matrix
    :param element_index: Index of element is being approximated
    :param xi: Coordinate in the first parametric direction
    :param eta: Coordinate in the second parametric direction
    :return: Tuple: jacobian, array of the shape functions, array of the shape functions derivatives in the first
    direction of computational domain, array of the shape functions derivatives in the second direction of computational
    domain
    """
    from numpy import array
    from numpy import sum
    from numpy.linalg import det
    from numpy.linalg import inv
    shape = array([
        1.0 - xi - eta,
        xi,
        eta
    ])  # linear shape functions
    shape_dxi = array([
        -1.0,
        1.0,
        0.0
    ])  # derivatives of the shape functions in the first parametric direction
    shape_deta = array([
        -1.0,
        0.0,
        1.0
    ])
    x = nodes[elements[element_index, :], 0]
    y = nodes[elements[element_index, :], 1]
    jacobi = array([
        [sum(shape_dxi * x), sum(shape_dxi * y)],
        [sum(shape_deta * x), sum(shape_deta * y)]
    ])  # Jacobi matrix
    jacobian = det(jacobi)
    inverted_jacobi = inv(jacobi)
    shape_dx = inverted_jacobi[0, 0] * shape_dxi + inverted_jacobi[0, 1] * shape_deta
    shape_dy = inverted_jacobi[1, 0] * shape_dxi + inverted_jacobi[1, 1] * shape_deta
    return jacobian, shape, shape_dx, shape_dy
