#!/usr/bin/env python
# -*- coding: utf-8 -*-


def legendre_interval(count):
    """
    Gauss-Legendre rules of the interval [-1; 1]
    :param count: Count of quadrature points
    :return tuple(array of points, array of weights). Both arrays are the same size.
    """
    from math import sqrt
    from numpy import array
    if count <= 1:
        p = array([0.0])
        w = array([2.0])
        return p, w
    elif count == 2:
        p = array([-1.0 / sqrt(3.0),
                   1.0 / sqrt(3.0)])
        w = array([1.0,
                   1.0])
        return p, w
    elif count == 3:
        p = array([-sqrt(3.0 / 5.0),
                   0.0,
                   sqrt(3.0 / 5.0)])
        w = array([5.0 / 9.0,
                   8.0 / 9.0,
                   5.0 / 9.0])
        return p, w
    elif count == 4:
        p = array([-sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0),
                   -sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0),
                   sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0),
                   sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0)])
        w = array([(18.0 - sqrt(30.0)) / 36.0,
                   (18.0 + sqrt(30.0)) / 36.0,
                   (18.0 + sqrt(30.0)) / 36.0,
                   (18.0 - sqrt(30.0)) / 36.0])
        return p, w
    else:
        p = array([-(1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)),
                   -(1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)),
                   0.0,
                   (1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)),
                   (1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0))])
        w = array([(322.0 - 13.0 * sqrt(70.0)) / 900.0,
                   (322.0 + 13.0 * sqrt(70.0)) / 900.0,
                   128.0 / 225.0,
                   (322.0 + 13.0 * sqrt(70.0)) / 900.0,
                   (322.0 - 13.0 * sqrt(70.0)) / 900.0])
        return p, w


def legendre_triangle(order):
    """
    Gauss-Legendre rules of the unit triangle
    :param order: Approximation order (1 - Linear, 2 - Quadratic, 3 - Cubic...)
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of weights). The arrays are the same size.
    """
    from math import sqrt
    from numpy import array
    if order <= 1:
        xi = array([1.0 / 3.0])
        eta = array([1.0 / 3.0])
        w = array([1.0 / 2.0])
        return xi, eta, w
    elif order == 2:
        xi = array([1.0 / 2.0,
                    0.0,
                    1.0 / 2.0])
        eta = array([1.0 / 2.0,
                     1.0 / 2.0,
                     0.0])
        w = array([1.0 / 6.0,
                   1.0 / 6.0,
                   1.0 / 6.0])
        return xi, eta, w
    elif order == 3:
        xi = array([1.0 / 6.0,
                    2.0 / 3.0,
                    1.0 / 6.0])
        eta = array([1.0 / 6.0,
                     1.0 / 6.0,
                     2.0 / 3.0])
        w = array([1.0 / 6.0,
                   1.0 / 6.0,
                   1.0 / 6.0])
        return xi, eta, w
    else:
        xi = array([1.0 / 3.0,
                    1.0 / 5.0,
                    1.0 / 5.0,
                    3.0 / 5.0])
        eta = array([1.0 / 3.0,
                     3.0 / 5.0,
                     1.0 / 5.0,
                     1.0 / 5.0])
        w = array([-9.0 / 32.0,
                   25.0 / 96.0,
                   25.0 / 96.0,
                   25.0 / 96.0])
        return xi, eta, w


def legendre_tetrahedra(order):
    """
    Gauss-Legendre rules of the unit tetrahedra
    :param order: Approximation order (1 - Linear, 2 - Quadratic, 3 - Cubic)
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of coordinates in the third parametric direction, array of weights). The arrays are the same size.
    """
    from math import sqrt
    from numpy import array
    if order <= 1:
        xi = array([0.25])
        eta = array([0.25])
        mu = array([0.25])
        w = array([1.0]) / 6.0
        return xi, eta, mu, w
    elif order == 2:
        xi = array([(5.0 + 3.0 * sqrt(5.0)) / 20.0,
                    (5.0 - sqrt(5.0)) / 20.0,
                    (5.0 - sqrt(5.0)) / 20.0,
                    (5.0 - sqrt(5.0)) / 20.0])
        eta = array([(5.0 - sqrt(5.0)) / 20.0,
                     (5.0 + 3.0 * sqrt(5.0)) / 20.0,
                     (5.0 - sqrt(5.0)) / 20.0,
                     (5.0 - sqrt(5.0)) / 20.0])
        mu = array([(5.0 - sqrt(5.0)) / 20.0,
                    (5.0 - sqrt(5.0)) / 20.0,
                    (5.0 + 3.0 * sqrt(5.0)) / 20.0,
                    (5.0 - sqrt(5.0)) / 20.0])
        w = array([0.25,
                   0.25,
                   0.25,
                   0.25]) / 6.0
        return xi, eta, mu, w
    else:
        xi = array([1.0 / 4.0,
                    1.0 / 2.0,
                    1.0 / 6.0,
                    1.0 / 6.0,
                    1.0 / 6.0])
        eta = array([1.0 / 4.0,
                     1.0 / 6.0,
                     1.0 / 2.0,
                     1.0 / 6.0,
                     1.0 / 6.0])
        mu = array([1.0 / 4.0,
                    1.0 / 6.0,
                    1.0 / 6.0,
                    1.0 / 2.0,
                    1.0 / 6.0])
        w = array([-4.0 / 5.0,
                   9.0 / 20.0,
                   9.0 / 20.0,
                   9.0 / 20.0,
                   9.0 / 20.0]) / 6.0
        return xi, eta, mu, w


def legendre_quad(count):
    """
    Gauss-Legendre rules of the quad [-1; 1] x [-1; 1]
    :param count: Count of quadrature points in each direction
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of weights). The arrays are the same size.
    """
    from numpy import zeros
    (p, w) = legendre_interval(count)
    size = len(p)
    xi = zeros(size * size)
    eta = zeros(size * size)
    weight = zeros(size * size)
    for i in range(size):
        for j in range(size):
            xi[i * size + j] = p[i]
            eta[i * size + j] = p[j]
            weight[i * size + j] = w[i] * w[j]
    return xi, eta, weight


def legendre_hexahedra(count):
    """
    Gauss-Legendre rules of the hexahedra [-1; 1] x [-1; 1] x [-1; 1]
    :param count: Count of quadrature points in each direction
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of coordinates in the third parametric direction, array of weights). The arrays are the same size.
    """
    from numpy import zeros
    (p, w) = legendre_interval(count)
    size = len(p)
    xi = zeros(size * size * size)
    eta = zeros(size * size * size)
    mu = zeros(size * size * size)
    weight = zeros(size * size * size)
    size_size = size * size
    for i in range(size):
        for j in range(size):
            for k in range(size):
                xi[i * size_size + j * size + k] = p[i]
                eta[i * size_size + j * size + k] = p[j]
                mu[i * size_size + j * size + k] = p[k]
                weight[i * size_size + j * size + k] = w[i] * w[j] * w[k]
    return xi, eta, mu, weight


def lobatto_interval(count):
    """
    Gauss-Lobatto rules of the interval [-1; 1]
    http://www.dam.brown.edu/people/alcyew/handouts/GLquad.pdf
    :param count: Count of quadrature points
    :return: tuple(array of points, array of weights). Both arrays are the same size.
    """
    from math import sqrt
    from numpy import array
    if count <= 2:
        p = array([-1.0,
                   1.0])
        w = array([1.0,
                   1.0])
        return p, w
    elif count == 3:
        p = array([-1.0,
                   0.0,
                   1.0])
        w = array([1.0 / 3.0,
                   4.0 / 3.0,
                   1.0 / 3.0])
        return p, w
    elif count == 4:
        p = array([-1.0,
                   -1.0 / sqrt(5.0),
                   1.0 / sqrt(5.0),
                   1.0])
        w = array([1.0 / 6.0,
                   5.0 / 6.0,
                   5.0 / 6.0,
                   1.0 / 6.0])
        return p, w
    else:
        p = array([-1.0,
                   -sqrt(3.0 / 7.0),
                   0.0,
                   sqrt(3.0 / 7.0),
                   1.0])
        w = array([1.0 / 10.0,
                   49.0 / 90.0,
                   32.0 / 45.0,
                   49.0 / 90.0,
                   1.0 / 10.0])
        return p, w


def lobatto_quad(count):
    """
    Gauss-Lobatto rules of the quad [-1; 1] x [-1; 1]
    :param count: Count of quadrature points in each direction
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of weights). The arrays are the same size.
    """
    from numpy import zeros
    (p, w) = lobatto_interval(count)
    size = len(p)
    xi = zeros(size * size)
    eta = zeros(size * size)
    weight = zeros(size * size)
    for i in range(size):
        for j in range(size):
            xi[i * size + j] = p[i]
            eta[i * size + j] = p[j]
            weight[i * size + j] = w[i] * w[j]
    return xi, eta, weight


def lobatto_hexahedra(count):
    """
    Gauss-Lobatto rules of the hexahedra [-1; 1] x [-1; 1] x [-1; 1]
    :param count: Count of quadrature points in each direction
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric
    direction, array of coordinates in the third parametric direction, array of weights). The arrays are the same size.
    """
    from numpy import zeros
    (p, w) = lobatto_interval(count)
    size = len(p)
    xi = zeros(size * size * size)
    eta = zeros(size * size * size)
    mu = zeros(size * size * size)
    weight = zeros(size * size * size)
    size_size = size * size
    for i in range(size):
        for j in range(size):
            for k in range(size):
                xi[i * size_size + j * size + k] = p[i]
                eta[i * size_size + j * size + k] = p[j]
                mu[i * size_size + j * size + k] = p[k]
                weight[i * size_size + j * size + k] = w[i] * w[j] * w[k]
    return xi, eta, mu, weight


if __name__ == "__main__":
    print("<=== Gauss-Legendre Rules Test ===>")
    print("Rules of the interval [-1; 1]")
    for i in range(1, 6):
        (p, w) = legendre_interval(i)
        print(i.__str__() + ": " + p.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the unit triangle")
    for i in range(1, 5):
        (xi, eta, w) = legendre_triangle(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the unit tetrahedra")
    for i in range(1, 4):
        (xi, eta, mu, w) = legendre_tetrahedra(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + mu.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the quad [-1; 1] x [-1; 1]")
    for i in range(2, 6):
        (xi, eta, w) = legendre_quad(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the hexahedra [-1; 1] x [-1; 1] x [-1; 1]")
    for i in range(2, 6):
        (xi, eta, mu, w) = legendre_hexahedra(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + mu.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("<================================>")
    print("<=== Gauss-Lobatto Rules Test ===>")
    print("Rules of the interval [-1; 1]")
    for i in range(2, 6):
        (p, w) = lobatto_interval(i)
        print(i.__str__() + ": " + p.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the quad [-1; 1] x [-1; 1]")
    for i in range(2, 6):
        (xi, eta, w) = lobatto_quad(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the hexahedra [-1; 1] x [-1; 1] x [-1; 1]")
    for i in range(2, 6):
        (xi, eta, mu, w) = lobatto_hexahedra(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + mu.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())