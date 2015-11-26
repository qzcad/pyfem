#!/usr/bin/env python
# -*- coding: utf-8 -*-


def plot_coo_matrix(m):
    """
    Subroutine plots image of sparse matrix
    :param m: Sparse matrix in the COO format (otherwise it will be converted to the COO-sparse format)
    :return: nothing
    """
    import matplotlib.pyplot as plt
    from scipy.sparse import coo_matrix
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, '.', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    import numpy as np
    from scipy.sparse import coo_matrix
    shape = (100000, 100000)
    rows = np.int_(np.round_(shape[0]*np.random.random(1000)))
    cols = np.int_(np.round_(shape[1]*np.random.random(1000)))
    vals = np.ones_like(rows)

    m = coo_matrix((vals, (rows, cols)), shape=shape)
    plot_coo_matrix(m)
