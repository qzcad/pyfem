#!/usr/bin/env python
# -*- coding: utf-8 -*-


def plane_strain_isotropic(e, nu):
    """
    Function builds the stress-strain relations matrix for isotropic material (plane strain case)
    :param e: Young's modulus
    :param nu: Poisson's ratio
    :return: The stress-strain relations matrix for isotropic material
    """
    from numpy import array
    d = array([
        [1.0 - nu,  nu,         0.0],
        [nu,        1.0 - nu,   0.0],
        [0.0,       0.0,        (1.0 - 2.0 * nu) / 2.0]
    ]) * e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return d


def plane_stress_isotropic(e, nu):
    """
    Function builds the stress-strain relations matrix for isotropic material (plane stress case)
    :param e: Young's modulus
    :param nu: Poisson's ratio
    :return: The stress-strain relations matrix for isotropic material
    """
    from numpy import array
    d = array([
        [1.0,   nu,     0.0],
        [nu,    1.0,    0.0],
        [0.0,   0.0,    (1.0 - nu) / 2.0]
    ]) * e / (1.0 - nu * nu)
    return d


if __name__ == "__main__":
    print(plane_strain_isotropic(203200.0, 0.27))
    print(plane_stress_isotropic(203200.0, 0.27))
