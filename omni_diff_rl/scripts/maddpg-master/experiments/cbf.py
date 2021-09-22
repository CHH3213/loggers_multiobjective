# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

def fun(args):
    a, b = args
    v = lambda x: (x[1] - b)**2
    return v


def con1(args):
    vmax, v, wp = args
    wmin = -8*(vmax - np.abs(v))
    wmax = 8*(vmax - np.abs(v))
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + vmax}, \
            {'type': 'ineq', 'fun': lambda x: -x[0] + vmax}, \
            {'type': 'ineq', 'fun': lambda x: x[1] - wmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[1] + wmax},\
            {'type': 'ineq', 'fun': lambda x: x[1] - wp - 0.5})
    return cons


def con2(args):
    vmax, v, wp = args
    wmin = -8*(vmax - np.abs(v))
    wmax = 8*(vmax - np.abs(v))
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + vmax}, \
            {'type': 'ineq', 'fun': lambda x: -x[0] + vmax}, \
            {'type': 'ineq', 'fun': lambda x: x[1] - wmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[1] + wmax},\
            {'type': 'ineq', 'fun': lambda x: -x[1] + wp - 0.5})
    return cons


def cbf(obs, action):
    ve_max = 1.5
    b = 0.125
    args = action[1]
    args1 = (1.5, np.sqrt(obs[1][2]**2 + obs[1][3]**2), obs[0][5]) # 原
    cons1 = con1(args1)
    cons2 = con2(args1)
    dist = np.sqrt(np.square(obs[0][0])+np.square(obs[0][1]))
    x0 = np.asarray((0, 0))
    if dist < 1.5:
        dtheta = obs[1][4] - obs[0][4]
        if dtheta < -3.14:
            dtheta = dtheta + 6.28
        elif dtheta > 3.14:
            dtheta = dtheta - 6.28
        if dtheta > 0:
            res = minimize(fun(args), x0, method='SLSQP', constraints=cons2)
            action[1] = res.x
            action[1][0] = 1.4
        else:
            res = minimize(fun(args), x0, method='SLSQP', constraints=cons1)
            action[1] = res.x
            action[1][0] = 1.4
    else:
        action = action
    return action
