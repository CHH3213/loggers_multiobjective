# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import numpy as np
def default_e(obs_n,action_n) :
    pos = [0, 0, 0]
    pos[0] = obs_n[1][3] #delta_x
    pos[1] = obs_n[1][4] #delta_y
    pos[2] = obs_n[1][2] #欧拉角
    ct = np.sqrt(np.square(pos[0]) + np.square((pos[1])))
    cost = pos[0] / ct
    theta = np.arccos(cost)  #相对位置夹角
    if pos[1] > 0:
        theta = -theta
    elif pos[1] <= 0:
        theta = theta
    pos[2] = theta - pos[2]
    if 0<pos[2] < 3.14/2:
        pos[2] -= 3.14-theta
    elif 0>pos[2] > -3.14/2:
        pos[2] += 3.14+theta
    # if pos[2] < 0:
    #     pos[2] = -1
    # if pos[2] > 0:
    #     pos[2] = 1
    # if pos[2] == 0:
    #     pos[2] = 0
    if ct<5:
        action_n[1] = [-pos[2]]
    return action_n[1]