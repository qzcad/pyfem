#!/usr/bin/env python
# -*- coding: utf-8 -*-


def linear_curve(t, p0, p1):
    """
    Linear interpolation of two points
    :param t: A value from the segment [0; 1]
    :param p0: First point
    :param p1: Second point
    :return: Point on the line p0p1
    """
    return p0 + t * (p1 - p0)


def quadratic_bezier_curve(t, p0, p1, p2):
    """
    Quadratic Bezier curve
    :param t: A value from the segment [0; 1]
    :param p0: First point
    :param p1: Second point
    :param p2: Third point
    :return: Point on a curve that defined by the points p0, p1, p2
    """
    return (1.0 - t) ** 2.0 * p0 + 2.0 * (1.0 - t) * t * p1 + t ** 2.0 * p2


def cubic_bezier_curve(t, p0, p1, p2, p3):
    """
    Cubic Bezier curve
    :param t: A value from the segment [0; 1]
    :param p0: First point
    :param p1: Second point
    :param p2: Third point
    :param p3: Fourth point
    :return: Point on a curve that defined by the points p0, p1, p2, p3
    """
    return (1.0 - t) ** 3.0 * p0 + 3.0 * (1.0 - t) ** 2.0 * t * p1 + 3.0 * (1.0 - t) * t * t * p2 + t ** 3.0 * p3


if __name__ == "__main__":
    from numpy import array
    p0 = array([0.0, 0.0])
    p1 = array([1.0, 2.0])
    print (linear_curve(0.5, p0, p1))
    p0 = array([-1.0, 0.0])
    p1 = array([0.0, 0.0])
    p2 = array([1.0, 0.0])
    print (quadratic_bezier_curve(0.5, p0, p1, p2))
    p0 = array([1.0, 1.0])
    p1 = array([0.0, 2.0])
    p2 = array([0.0, 2.5])
    p3 = array([1.0, 3.0])
    print (cubic_bezier_curve(0.5, p0, p1, p2, p3))