import numpy as np
from scipy.optimize import minimize

def fun(args):
    b = args
    v = lambda x: (x - b)**2
    return v

def Nominal_control(state):
    '''
    as a nominal controller
    '''
    is_poistive = 1
    distance = np.sqrt(np.square(state[0][2])+np.square(state[0][3]))
    theta_e = np.arctan2(state[0][3],state[0][2])  - state[0][4]

    if(theta_e>np.pi):
        theta_e -= 2*np.pi
    elif (theta_e<-np.pi):
        theta_e += 2*np.pi
    # print(theta_e)
    
    if(theta_e>np.pi/2 or theta_e<-np.pi/2):
        is_poistive = -1

    w = theta_e*is_poistive
    return w

'''first cbf'''
def con1(args,A1,A2_1,A2_2,A3,A4):
    wmin,wmax = args
    cons = (
            {'type': 'ineq', 'fun': lambda x: x - wmin}, \
            {'type': 'ineq', 'fun': lambda x: -x + wmax},\
            {'type': 'ineq', 'fun': lambda x: A1+2*(A2_1+x*A2_2)+A3+A4})
    return cons

def ECBF(state, action):
    # state[0]is state of pursuer

    v_p, v_e= 1, 0.787
    w_max = 2
    d = np.sqrt(np.square(state[0][2])+np.square(state[0][3]))
    r = 0.3
    k1,k2 = 1,1
    delta_p = np.array(state[0][2:4])
    delta_v = np.array([v_p*np.cos(state[0][5])-v_e*np.cos(state[1][5]),v_p*np.sin(state[0][5])-v_e*np.sin(state[1][5])])
    delta_a = lambda x:np.array([-v_p*w_max*np.sin(state[0][5])+v_e*x*np.sin(state[1][5]),v_p*w_max*np.cos(state[0][5])-v_e*x*np.cos(state[1][5])])
    h_ = d**2-r**2
    h_dot1 = np.sum(2*delta_p*delta_v)
    h_dot2 = 2*(np.square(delta_v)+np.sum(delta_p*delta_a))
    p1 = -h_dot1/h_+0.5  #pole
    p2 = -(h_dot2+p1*h_dot1)/(h_dot1+p1*h_)+0.5  #pole
    A1 = 2*(v_p**2+v_e**2-2*v_p*v_e*np.cos(state[0][4]))
    A2_1 = v_p*w_max*(-state[1][2]*np.sin(state[0][5]))+state[1][3]*np.cos(state[0][5])
    A2_2 = v_e * (state[1][2]*np.sin(state[1][5])-state[1][3]*np.cos(state[1][5]))
    A3 = 2*k1*(v_p*(state[1][2]*np.cos(state[0][5]+state[1][3]*np.sin(state[0][5])))-v_e*(state[1][2]*np.cos(state[1][5])+state[1][3]*np.sin(state[1][5])))
    A4 = k2*(d**2-r**2)

    # w = Nominal_control(state)
    w = action[1]
    args = w
    args1 = (-2,2)
    cons1 = con1(args1,A1,A2_1,A2_2,A3,A4)
    x0 = np.asarray((0))
    if d < 1:
        
        res = minimize(fun(args), x0, method='SLSQP', constraints=cons1)
        action[1] = res.x
        # print(res.success)
    return action

'''second cbf'''
def con2(args,delta_theta,h_x):
    wmin,wmax = args
    cons = (
            {'type': 'ineq', 'fun': lambda x: x - wmin}, \
            {'type': 'ineq', 'fun': lambda x: -x + wmax},\
            {'type': 'ineq', 'fun': lambda x: -np.sin(delta_theta)*x+h_x})
    return cons

def ZCBF(state,action):
    delta_p = np.array(state[0][2:4])
    delta_theta = np.array(state[1][4])
    delta_vmax = 1-0.787
    d = np.linalg.norm(delta_p)
    k = 0.1
    h_x = np.linalg.norm(delta_p)**2+np.cos(delta_theta)-k*delta_vmax
    # w = Nominal_control(state)
    w = action[1]
    args = (-2,2)
    x0 = np.asarray((0))
    cons2 = con2(args,delta_theta,h_x)
    if d < 1:
        
        res = minimize(fun(w), x0, method='SLSQP', constraints=cons2)
        action[1] = res.x
        # print(res.success)
    return action

'''third cbf'''
def con3(args,phi0,phi1,phi2):
    wmin,wmax = args
    cons = (
            {'type': 'ineq', 'fun': lambda x: x - wmin}, \
            {'type': 'ineq', 'fun': lambda x: -x + wmax},\
            {'type': 'ineq', 'fun': phi0},\
            {'type': 'ineq', 'fun': phi1},\
            {'type': 'ineq', 'fun':  lambda x: phi2}
            )
    return cons
def HOCBF(state,action):
    
    v_p, v_e= 1, 0.787
    w_max = 2
    d = np.sqrt(np.square(state[0][2])+np.square(state[0][3]))
    r = 0.3
    delta_p = np.array(state[0][2:4])
    delta_v = np.array([v_p*np.cos(state[0][5])-v_e*np.cos(state[1][5]),v_p*np.sin(state[0][5])-v_e*np.sin(state[1][5])])
    delta_a = lambda x:np.array([-v_p*w_max*np.sin(state[0][5])+v_e*x*np.sin(state[1][5]),v_p*w_max*np.cos(state[0][5])-v_e*x*np.cos(state[1][5])])
    h_ = np.sum(np.square(delta_p))-r**2
    h_dot1 = np.sum(2*delta_p*delta_v)
    h_dot2 = lambda x: 2*(np.sum(np.square(delta_v))+np.sum(delta_p*delta_a))
    phi_0 = h_
    phi_1 = h_dot1 + h_
    phi_2 = lambda x:h_dot2+2*h_dot1+h_
    args = (-2,2)
    x0 = np.asarray((0))
    cons = con3(args,phi_0,phi_1,phi_2)
    w = action[1]
    if d < 100:
        
        res = minimize(fun(w), x0, method='SLSQP', constraints=cons)
        action[1] = res.x
        print(res.success)
    return action


def con4(state,action,delta_p,delta_v,h_,h_dot1):
    v_p, v_e= 1, 0.787
    w_max = 2
    # h_dot2 = lambda x: 2*(np.sum(np.square(delta_v))+np.sum(delta_p*delta_a))
    # phi_0 = h_
    # phi_1 = h_dot1 + h_
    # phi_2 = lambda x:h_dot2+2*h_dot1+h_
    cons = (
        {'type': 'ineq', 'fun': lambda x: x + w_max}, \
        {'type': 'ineq', 'fun': lambda x: -x + w_max},\
        {'type': 'ineq', 'fun':  lambda x: 2*h_dot1+h_+2*(np.sum(np.square(delta_v))+np.sum(delta_p*np.array([-v_p*action[0]*np.sin(state[0][5])+v_e*x*np.sin(state[1][5]),v_p*action[0]*np.cos(state[0][5])-v_e*x*np.cos(state[1][5])])))})
    return cons

def HOCBF2(state,action):
    x0 = np.asarray((1))
    w = action[1]
    v_p, v_e= 1.4, 1.24
    w_max = 2
    d = np.sqrt(np.square(state[0][2])+np.square(state[0][3]))
    r = 0.3
    delta_p = np.array(state[0][2:4])
    delta_v = np.array([v_p*np.cos(state[0][5])-v_e*np.cos(state[1][5]),v_p*np.sin(state[0][5])-v_e*np.sin(state[1][5])])
    # delta_a = lambda x:np.array([-v_p*w_max*np.sin(state[0][5])+v_e*x*np.sin(state[1][5]),v_p*w_max*np.cos(state[0][5])-v_e*x*np.cos(state[1][5])])
    h_ = np.sum(np.square(delta_p))-r**2
    h_dot1 = np.sum(2*delta_p*delta_v)
    cons = con4(state,action,delta_p,delta_v,h_,h_dot1)

    # if  h_<0 or h_dot1<0:
        
    res = minimize(fun(w), x0, method='SLSQP', constraints=cons)
    action[1] = res.x
    # print(res.success)
    # print(res.x)
    return action




if __name__ == '__main__':
    a1 = np.array([1,2,3])
    a2 = np.array([1,2,3])
    print(np.square(a1*a2))
    state = [[1,2,3,4,5,6],
    [6.5,5,4,3,20,1]]
    action = [-1,0.5]
    a = HOCBF2(state ,action)
    print(a)

