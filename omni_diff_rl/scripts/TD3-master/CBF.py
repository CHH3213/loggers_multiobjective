import numpy as np
from scipy.optimize import minimize


def QPfun(ud):
    def fun(u):
        return (u[0] - ud[0])**2 / 2
    return fun


def constrains(State):
    '''
    State[0] = Xp
    State[1] = Yp
    State[2] = th_p (rad)
    State[3] = Xe
    State[4] = Ye
    State[5] = th_e (rad)
    State[6] = wp
    '''
    Vp = 1.0
    Ve = 0.787
    r = 0.3

    Xp, Yp, th_p, Xe, Ye, th_e, wp = State
    sinp, cosp, sine, cose = np.sin(th_p), np.cos(
        th_p), np.sin(th_e), np.cos(th_e)
    Dp = np.array([Xp-Xe, Yp-Ye])
    Dv = np.array([Vp*cosp-Ve*cose, Vp*sinp-Ve*sine])
    Dvp = np.array([-Vp*sinp, Vp*cosp])
    Dve = np.array([-Ve*sine, Ve*cose])

    K0 = 0.8
    K1 = 0.4

    def con_we(we):
       
        return 2*(Vp**2 + Ve**2 - 2*Vp*Ve*np.cos(th_p-th_e)) + 2*wp*np.einsum('i,i->', Dp, Dvp) - \
        2*we*np.einsum('i,i->', Dp, Dve) + \
        K0*(np.einsum('i,i->', Dp, Dp) - r**2) + \
        2*K1*(np.einsum('i,i->', Dp, Dv))


    cons = (
        {'type': 'ineq', 'fun': con_we},
        {'type': 'ineq', 'fun': lambda u: u[0]+2},
        {'type': 'ineq', 'fun': lambda u: 2-u[0]}
    )

    return cons


def CBF(u, State):
    '''
    u=[we]

    State[0] = Xp
    State[1] = Yp
    State[2] = th_p (rad)
    State[3] = Xe
    State[4] = Ye
    State[5] = th_e (rad)
    State[6] = wp
    '''
    x0 = u
    wmax,wmin = 2,-2
    if State[6]<0:
        State[6]=wmax
    elif State[6]>0:
        State[6]=wmin
    else:
        State[6] = 0

    res = minimize(fun=QPfun(u), x0=x0, constraints=constrains(State))
    # print(res.success)
    return res.x


if __name__ == '__main__':
    pass
