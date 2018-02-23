#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of variogram models.
"""
from __future__ import division, absolute_import, print_function

from functools import partial
import numpy as np


def linear(params, x):
    return params[0] * x + params[1]

def gau(params, x):
    return params[0] * np.exp(- x**2 / params[1]**2)

def exp(params, x):
    return params[0] * np.exp(- x / params[1])
