#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kriging methods.
"""
from __future__ import division, absolute_import, print_function

from functools import partial
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist, pdist
from gstools.variogram import models


class Kriging:
    """A kriging class.
        
    Args:
        dim (int): spatial dimension
        pos (array/ tuple): position of data values, either an array or
                            a tuple of arrays for dim > 1
        data (array): a 1-dim array containing the data to be kriged
    """
    def __init__(self, dim, pos, data, mesh_type='structured'):
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
        if dim > 1:
            if mesh_type != 'structured' and mesh_type != 'unstructured':
                raise ValueError('Unknown mesh type {0}'.format(mesh_type))
            if len(pos) != dim:
                raise ValueError('For {0} dimenions, provide a {0} position tuple'.format(dim))
            for i in range(dim):
                if len(pos[i]) != len(data):
                    raise ValueError('Position tuple and data array must be equally long')
            self._pos_data = np.stack(pos, axis=1)
        else:
            if len(pos) != len(data):
                raise ValueError('Position and data arrays must be equally long')
            self._pos_data = pos
        self._data = data
        self._mesh_type = mesh_type

    def _kriging_matrix(self, model_func):
        """Creates the kriging matrix A for given variogram model.

            ( gamma(x1,x1) gamma(x1,x2) ... gamma(x1,xn) 1 )
            ( gamma(x2,x1) gamma(x2,x2) ... gamma(x2,xn) 1 )
            (      .            .       ...      .       . )
        A = (      .            .       ...      .       . )
            (      .            .       ...      .       . )
            ( gamma(xn,x1) gamma(xn,x2) ... gamma(xn,xn) 1 )
            (      1            1       ...      1       0 )
        """
        n = self._pos_data.shape[0]
        if self._dim == 1:
            self._pos_data = np.stack((self._pos_data, self._pos_data), axis=1)
        dist = cdist(self._pos_data, self._pos_data)
        A = np.ones((n+1, n+1))
        A[:n, :n] = -model_func(dist)
        np.fill_diagonal(A, 0.)
        return A

    def ordinary(self, pos_eval, model='gau', **kwargs):
        #no parameters for model provided, thus fitting
        if not kwargs:
            raise NotImplementedError('Variogram parameter fitting not yet implemented.')
        if model == 'linear':
            params = [kwargs['slope'], kwargs['nugget']]
            model_func = partial(models.linear, params)

        if self._dim == 1:
            Z, var = self.ordinary_1d(pos_eval, model_func)
        elif self._dim == 2:
            Z, var = getattr(self, 'ordinary_{}_2d'.format(self._mesh_type))(pos_eval, model_func)
        return Z, var

    def ordinary_1d(self, pos_eval, model_func):
        ng = len(pos_eval)
        A = self._kriging_matrix(model_func)
        A_inv = sp.linalg.inv(A)

        n = self._pos_data.shape[0]

        pos_eval = np.stack((pos_eval, pos_eval), axis=1)

        b_dist = cdist(pos_eval, self._pos_data)
        b = np.ones((b_dist.shape[0], n+1))
        b[:,:n] = -model_func(b_dist)
        lambd = np.matmul(A_inv, b.T).T
        z_values = np.sum(lambd[:,:n] * self._data, axis=1)
        var = np.sum(lambd * -b, axis=1)

        return z_values, var

    def ordinary_structured_2d(self, pos_eval, model_func):
        ng = []
        for d in range(self._dim):
            ng.append(len(pos_eval[d]))
        grid_x, grid_y = np.meshgrid(pos_eval[0], pos_eval[1])
        pos_eval = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
        z_values, var = self.ordinary_unstructured_2d(pos_eval, model_func)
        z_values = z_values.reshape((ng[1], ng[0]))
        var = var.reshape((ng[1], ng[0]))
        return z_values, var

    def ordinary_unstructured_2d(self, pos_eval, model_func):
        A = self._kriging_matrix(model_func)
        A_inv = sp.linalg.inv(A)

        n = self._pos_data.shape[0]
        # the distances between the positions of the data and
        # the evaluation positions
        b_dist = cdist(pos_eval, self._pos_data)
        # the kriging vector b with variograms of data pos and eval pos
        # for one pos.:
        # b1 = (gamma(x1,x1*), gamma(x2,x1*), ... gamma(xn,x1*,1))T
        b = np.ones((b_dist.shape[0], n+1))
        b[:,:n] = -model_func(b_dist)

        # the Lagrangian multipliers
        lambd = np.matmul(A_inv, b.T).T

        # the interpolated values
        z_values = np.sum(lambd[:,:n] * self._data, axis=1)
        # the variances
        var = np.sum(lambd * -b, axis=1)

        return z_values, var
