# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numba import jit


@jit(nopython=True)
def _distance_1d(x, y, z, i, j):
    return np.sqrt((x[i] - x[j]) * (x[i] - x[j]))


@jit(nopython=True)
def _distance_2d(x, y, z, i, j):
    return np.sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]))


@jit(nopython=True)
def _distance_3d(x, y, z,
                 i, j):
    return np.sqrt((x[i] - x[j]) * (x[i] - x[j]) +
                   (y[i] - y[j]) * (y[i] - y[j]) +
                   (z[i] - z[j]) * (z[i] - z[j]))


@jit
def unstructured(f, bin_edges, x, y=None, z=None):
    if x.shape[0] != f.shape[0]:
        raise ValueError('len(x) = {0} != len(f) = {1} '.
                         format(x.shape[0], f.shape[0]))
    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    # 3d
    if z is not None:
        if z.shape[0] != f.shape[0]:
            raise ValueError('len(z) = {0} != len(f) = {1} '.
                             format(z.shape[0], f.shape[0]))
        distance = _distance_3d
    # 2d
    elif y is not None:
        if y.shape[0] != f.shape[0]:
            raise ValueError('len(y) = {0} != len(f) = {1} '.
                             format(y.shape[0], f.shape[0]))
        distance = _distance_2d
    # 1d
    else:
        distance = _distance_1d

    i_max = bin_edges.shape[0] - 1
    j_max = x.shape[0] - 1
    k_max = x.shape[0]

    variogram = np.zeros(len(bin_edges)-1)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(x, y, z, k, j)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    counts[i] += 1
                    variogram[i] += (f[k] - f[j])**2
    # avoid division by zero
    for i in range(i_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def structured_3d(x, y, z, f):
    i_max = f.shape[0] - 1
    j_max = f.shape[1]
    k_max = f.shape[2]
    l_max = i_max + 1

    variogram = np.zeros(l_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                for l in range(1, l_max-i):
                    counts[l] += 1
                    variogram[l] += (f[i, j, k] - f[i+l, j, k])**2
    # avoid division by zero
    for i in range(l_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def structured_2d(x, y, f):
    i_max = f.shape[0] - 1
    j_max = f.shape[1]
    k_max = i_max + 1

    variogram = np.zeros(k_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(j_max):
            for k in range(1, k_max-i):
                counts[k] += 1
                variogram[k] += (f[i, j] - f[i+k, j])**2
    # avoid division by zero
    for i in range(k_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def structured_1d(x, f):
    i_max = f.shape[0] - 1
    j_max = i_max + 1

    variogram = np.zeros(j_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(1, j_max-i):
            counts[j] += 1
            variogram[j] += (f[i] - f[i+j])**2
    # avoid division by zero
    for i in range(j_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def ma_structured_3d(x, y, z, f, mask):
    i_max = f.shape[0] - 1
    j_max = f.shape[1]
    k_max = f.shape[2]
    l_max = i_max + 1

    variogram = np.zeros(l_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                for l in range(1, l_max-i):
                    if not mask[i, j, k] and not mask[i+l, j, k]:
                        counts[l] += 1
                        variogram[l] += (f[i, j, k] - f[i+l, j, k])**2
    # avoid division by zero
    for i in range(l_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def ma_structured_2d(x, y, f, mask):
    i_max = f.shape[0] - 1
    j_max = f.shape[1]
    k_max = i_max + 1

    variogram = np.zeros(k_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(j_max):
            for k in range(1, k_max-i):
                if not mask[i, j] and not mask[i+k, j]:
                    counts[k] += 1
                    variogram[k] += (f[i, j] - f[i+k, j])**2
    # avoid division by zero
    for i in range(k_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


@jit
def ma_structured_1d(x, f, mask):
    i_max = f.shape[0] - 1
    j_max = i_max + 1

    variogram = np.zeros(j_max)
    counts = np.zeros_like(variogram, dtype=np.int)

    for i in range(i_max):
        for j in range(1, j_max-i):
            if not mask[i] and not mask[j]:
                counts[j] += 1
                variogram[j] += (f[i] - f[i+j])**2
    # avoid division by zero
    for i in range(j_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)
