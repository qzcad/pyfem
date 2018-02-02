#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import zeros, array, cross, copy
from numpy.linalg import norm
from scipy.sparse import lil_matrix

from mesh2d import read
from plot_coo_matrix import plot_coo_matrix
from print_progress import print_progress
from quadrature import legendre_quad, legendre_triangle
from shape_functions import iso_quad, iso_triangle
from stress_strain_matrix import plane_stress_isotropic


def cosine(a, b, c):
    """
    Routine builds the direction cosine matrix for the triangle
    :param a: Coordinates of the first node
    :param b: Coordinates of the second node
    :param c: Coordinates of the third node
    :return: The direction cosine matrix
    """
    ab = b - a
    ac = c - a
    n = cross(ab, ac)
    vx = ab / norm(ab, ord=2)
    vz = n / norm(n, ord=2)
    vy = cross(vz, vx)
    # print n, n[0] / norm(n, ord=2), n[1] / norm(n, ord=2), n[2] / norm(n, ord=2), norm(n, ord=2)
    l = array([
        [vx[0], vx[1], vx[2]],
        [vy[0], vy[1], vy[2]],
        [vz[0], vz[1], vz[2]]
    ])
    return l

def assembly_mindlin_shell(nodes, elements, thickness, elasticity_matrix, gauss_order=3, kappa=5.0/6.0):
    # type: (array, array, float, float, float, int, float) -> lil_matrix
    """
    Assembly Routine for the Mindlin Shells Analysis
    :param nodes: A three-dimensional array of plate's nodes coordinates
    :param elements: A two-dimensional array of plate's elements (mesh)
    :param thickness: The thickness of the shell
    :param elasticity_matrix: A two-dimensional array that represents stress-strain relations
    :param gauss_order: An order of gaussian quadratures
    :param kappa: The shear correction factor
    :return: Global stiffness matrix in the CSR sparse format
    Order: u_0, v_0, w_0, th_x_0, th_y_0, th_z_0, ..., u_(n-1), v_(n-1), w_(n-1), th_x_(n-1), th_y_(n-1), th_z_(n-1), ; n - nodes count
    """
    print "Assembly routine is started"
    freedom = 6  # degree of freedom
    element_nodes = len(elements[0])  # count of nodes per element
    nodes_count = len(nodes)  # count of nodes in the mesh
    elements_count = len(elements)  # count of elements in the mesh
    dimension = freedom * nodes_count  # a dimension of the problem
    element_dimension = freedom * element_nodes
    global_matrix = lil_matrix((dimension, dimension))
    xi = zeros(gauss_order)
    eta = zeros(gauss_order)
    w = zeros(gauss_order)

    if element_nodes == 4:
        (xi, eta, w) = legendre_quad(gauss_order)
    else:
        (xi, eta, w) = legendre_triangle(gauss_order)

    df = elasticity_matrix
    dc = array([
        [df[2, 2], 0.0],
        [0.0, df[2, 2]]
    ])

    for element_index in range(elements_count):
        local = zeros((element_dimension, element_dimension))
        global_coord = nodes[elements[element_index, :], :]
        # print element
        a = global_coord[0, :]
        b = global_coord[1, :]
        c = global_coord[2, :]
        la = cosine(a, b, c)
        T = zeros((element_dimension, element_dimension))
        for i in range(0, element_dimension, 3):
            T[i:i+3, i:i+3] = la

        local_coord = copy(global_coord)
        for i in range(element_nodes):
            local_coord[i, :] = la.dot(global_coord[i, :] - a)

        for i in range(len(w)):
            jacobian = 0.0
            shape = zeros(element_nodes)
            shape_dx = zeros(element_nodes)
            shape_dy = zeros(element_nodes)
            Bm = zeros((3, element_dimension))
            Bf = zeros((3, element_dimension))
            Bc = zeros((2, element_dimension))
            if element_nodes == 4:
                (jacobian, shape, shape_dx, shape_dy) = iso_quad(local_coord, xi[i], eta[i])
            else:
                (jacobian, shape, shape_dx, shape_dy) = iso_triangle(local_coord, xi[i], eta[i])

            for j in range(element_nodes):
                Bm[0, j * freedom] = shape_dx[j]
                Bm[1, j * freedom + 1] = shape_dy[j]
                Bm[2, j * freedom] = shape_dy[j]
                Bm[2, j * freedom + 1] = shape_dx[j]
                Bf[0, j * freedom + 3] = shape_dx[j]
                Bf[1, j * freedom + 4] = shape_dy[j]
                Bf[2, j * freedom + 3] = shape_dy[j]
                Bf[2, j * freedom + 4] = shape_dx[j]
                Bc[0, j * freedom + 2] = shape_dx[j]
                Bc[0, j * freedom + 3] = shape[j]
                Bc[1, j * freedom + 2] = shape_dy[j]
                Bc[1, j * freedom + 4] = shape[j]

            local = local + jacobian * w[i] * thickness * (Bm.transpose().dot(df).dot(Bm))
            local = local + jacobian * w[i] * (thickness ** 3.0) / 12.0 * (Bf.transpose().dot(df).dot(Bf))
            local = local + jacobian * w[i] * thickness * kappa * (Bc.transpose().dot(dc).dot(Bc))

        surf = T.transpose().dot(local).dot(T)
        for i in range(element_dimension):
            ii = elements[element_index, i / freedom] * freedom + i % freedom
            for j in range(i, element_dimension):
                jj = elements[element_index, j / freedom] * freedom + j % freedom
                global_matrix[ii, jj] += surf[i, j]
                if i != j:
                    global_matrix[jj, ii] = global_matrix[ii, jj]

        print_progress(element_index, elements_count - 1)

    print "\nThe assembly routine is completed"
    return global_matrix.tocsr()


if __name__ == "__main__":
    (nodes, quads) = read('examples/nose_cone_2017.txt')
    e = 1  # The Young's modulus
    nu = 0.3  # The Poisson's ratio
    df = plane_stress_isotropic(e, nu)
    stiffness = assembly_mindlin_shell(nodes=nodes, elements=quads, thickness=0.03, elasticity_matrix=df)
    #plot_coo_matrix(stiffness)

    r1 = 0.536 / 2.0
    r2 = 0.986 / 2.0
    r3 = 1.678 / 2.0
    r4 = 2.329 / 2.0
    r5 = 2.4 / 2.0
    def force_func(node):
        if abs(c - node[1]) < 0.0000001:
            return array([0.0, -q])
        return array([0.0, 0.0])