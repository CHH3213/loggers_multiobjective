# -*-coding:utf-8-*-
import numpy as np
def default_p(obs_n,action_n) :
    pos = [0, 0, 0]
    pos[0] = obs_n[0][3] #delta_x
    pos[1] = obs_n[0][4] #delta_y
    pos[2] = obs_n[0][2] #欧拉角
    ct = np.sqrt(np.square(pos[0]) + np.square((pos[1])))
    ct = pos[0] / ct
    theta = np.arccos(ct)
    if pos[1] > 0:
        theta = theta
    elif pos[1] <= 0:
        theta = -theta
    pos[2] = theta - pos[2]
    if pos[2] > 3.142:
        pos[2] -= 6.28
    elif pos[2] < -3.142:
        pos[2] += 6.28
    if pos[2] < 0:
        pos[2] = -1
    if pos[2] > 0:
        pos[2] = 1
    if pos[2] == 0:
        pos[2] = 0
    action_n[0] = [-pos[2]]
    return action_n[0]
