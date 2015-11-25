#!/usr/bin/env python
# -*- coding: utf-8 -*-


def legendre_line(count):
    """
    Gauss-Legendre rules of the interval [-1; 1]
    :param count: Count of quadrature points
    :return tuple(array of points, array of weights). Both arrays are the same size
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
    elif count >= 5:
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


def legendre_triangle(count):
    """
    Gauss-Legendre rules of the unit triangle
    :param count: Count of quadrature points
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric direction, array of weights). The arrays are the same size
    """
    from math import sqrt
    from numpy import array
    if count <= 1:
        xi = array([1.0 / 3.0])
        eta = array([1.0 / 3.0])
        w = array([1.0])
        return xi, eta, w
    elif count <= 3:  # there are not defined rules for two points in unit triangle
        xi = array([1.0 / 6.0,
                    2.0 / 3.0,
                    1.0 / 6.0])
        eta = array([1.0 / 6.0,
                     1.0 / 6.0,
                     2.0 / 3.0])
        w = array([1.0 / 3.0,
                   1.0 / 3.0,
                   1.0 / 3.0])
        return xi, eta, w
    elif count >= 4:
        xi = array([1.0 / 3.0,
                    1.0 / 5.0,
                    1.0 / 5.0,
                    3.0 / 5.0])
        eta = array([1.0 / 3.0,
                     3.0 / 5.0,
                     1.0 / 5.0,
                     1.0 / 5.0])
        w = array([-27.0 / 48.0,
                   25.0 / 48.0,
                   25.0 / 48.0,
                   25.0 / 48.0])
        return xi, eta, w


def legendre_quad(count):
    """
    Gauss-Legendre rules of the quad [-1; 1] x [-1; 1]
    :param count: Count of quadrature points in each direction
    :return: tuple(array of coordinates in the first parametric direction, array of coordinates in the second parametric direction, array of weights). The arrays are the same size
    """
    from numpy import zeros
    (p, w) = legendre_line(count)
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

# todo: Add Gauss–Lobatto Rules (http://mathworld.wolfram.com/LobattoQuadrature.html or http://www.dam.brown.edu/people/alcyew/handouts/GLquad.pdf)

if __name__ == "__main__":
    print("Gauss-Legendre Rules Test")
    print("Rules of the interval [-1; 1]")
    for i in range(1,6):
        (p, w) = legendre_line(i)
        print(i.__str__() + ": " + p.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the unit triangle")
    for i in (1, 3, 4):
        (xi, eta, w) = legendre_triangle(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())
    print("Rules of the quad [-1; 1] x [-1; 1]")
    for i in range(2,6):
        (xi, eta, w) = legendre_quad(i)
        print(i.__str__() + ": " + xi.__str__() + " " + eta.__str__() + " " + w.__str__() + " sum(w) = " + w.sum().__str__())