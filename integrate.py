#!/usr/bin/env python
# -*- coding: utf-8 -*-


def interval(func, a, b, count):
    """
    This routine integrates a function over an interval [a; b] using Gauss-Legendre rules
    :param a: Lower limit of the interval
    :param b: Upper limit of the interval
    :param func: A Python function or method to integrate
    :param count: Count of quadrature point
    :return:
    """
    from quadrature import legendre_interval
    from scipy.spatial.distance import euclidean
    (p, w) = legendre_interval(count)
    j = euclidean(a, b) / 2.0
    s = func(0.5 * a * (1.0 - p[0]) + 0.5 * b * (1.0 + p[0])) * w[0] * j
    for i in range(1, count):
        s += func(0.5 * a * (1.0 - p[i]) + 0.5 * b * (1.0 + p[i])) * w[i] * j
    return s


if __name__ == "__main__":
    from numpy import array

    def f(x): return 1.0

    a = array([0., 0.])
    b = array([1., 1.])
    print(interval(f, a, b, count=3))
