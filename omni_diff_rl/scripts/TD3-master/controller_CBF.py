from threading import Barrier
import numpy as np
from scipy.optimize import minimize

class CBF:
    def __init__(self,state,action_e):
        self.state_e = state[3:5]
        self.state_p = state[0:3]
        self.barrier = state[6:9]

        self.phi = action_e[0]
        # print(self.phi)
        # 实物对应训练
        # self.limie_phi = np.pi
        # self.v_e = 0.787*0.1
        # self.v_p = 1*0.1
        # self.v_barrier = 0.1*0.1
        # self.target_point = np.array([2.5,0])
        #
        # #仿真环境对应训练实物
        self.limie_phi = np.pi
        self.v_e = 0.787*0.2
        self.v_p = 1*0.2
        self.v_barrier = 0.07*0.1
        self.target_point = np.array([2.5,0])

        # self.limie_phi = np.pi
        # self.v_e = 0.787
        # self.v_p = 1
        # self.v_barrier = 0.3
        # self.target_point = np.array([6, 0])

        # 原设置
        self.target_judge = 0.1  
        self.barrier_judge = 0.4
        self.pv_judge = 0.5
        # 匹配真实环境
        # self.target_judge = 0.1
        # self.barrier_judge =0.4
        # self.pv_judge =0.8

    def QPfun(self,ud):
        def fun(u):
            return (u - ud)**2 / 2
        return fun   
    
    def barrier_cons(self,args):
        wmin,wmax = args
        # print(wmin,wmax)
        cons = (
                {'type': 'ineq', 'fun': lambda x: x - wmin}, \
                {'type': 'ineq', 'fun': lambda x: -x + wmax},\
                {'type': 'ineq', 'fun': lambda x: self.h_barrier-np.sum(2*self.delta_e_barrier*np.array([self.v_e*np.cos(x)-self.v_barrier,self.v_e*np.sin(x)]))/np.linalg.norm(self.delta_e_barrier)})
        return cons

    def barrier_CBF(self,nominal_control):
        '''
        nominal control is RL control
        nominal_control = [phi]
        return:action
        '''
        # print(self.barrier)
        self.delta_e_barrier = np.array(self.state_e)-np.array(self.barrier[0:2])
        self.h_barrier = np.linalg.norm(self.delta_e_barrier) - self.barrier_judge 
        x0 = nominal_control[0]
        args = (-self.limie_phi,self.limie_phi)
        barrier_cons = self.barrier_cons(args)
        if self.h_barrier <0.0:
            res = minimize(fun=self.QPfun( nominal_control[0]), method='SLSQP', x0=x0, constraints=barrier_cons)
            action =res.x
        else:
            action = self.phi

        # res = minimize(fun=self.QPfun( nominal_control[0]), x0=x0, constraints=barrier_cons)
        # action =res.x
        return action

    def pv_cons(self,args):
        wmin,wmax = args
        # print(wmin,wmax)
        cons = (
                {'type': 'ineq', 'fun': lambda x: x - wmin}, \
                {'type': 'ineq', 'fun': lambda x: -x + wmax},\
                {'type': 'ineq', 'fun': lambda x: (self.h_pv-np.sum(self.delta_pv*np.array([self.v_e*np.cos(x)-self.v_p*np.cos(self.state_p[2]),self.v_e*np.sin(x)-self.v_p*np.sin(self.state_p[2])]))/np.linalg.norm(self.delta_pv))})
        return cons
    def pv_CBF(self,nominal_control):
        '''
        nominal control is the optimal control
        nominal_control = [phi]
        return:action

        '''
        self.delta_pv = np.array(self.state_e)-np.array(self.state_p[0:2])
        self.h_pv = np.linalg.norm(self.delta_pv) - self.pv_judge 
        x0 = nominal_control[0]
        # x0 = 0
        args = (-self.limie_phi,self.limie_phi)
        pv_cons = self.pv_cons(args)
        if self.h_pv <0.0:
            res = minimize(fun=self.QPfun( nominal_control[0]), method='SLSQP', x0=x0, constraints=pv_cons)
            action =res.x
            # print(res.success)
            action = nominal_control[0]

        else:
            action =self.phi
        # res = minimize(fun=self.QPfun( nominal_control[0]), x0=x0, constraints=pv_cons)
        # action =res.x
        # print(res.success)
        # action = nominal_control[0]
        # print(self.h_pv)
        return action


    def composite_cons(self,args):
        wmin, wmax = args
        # print(wmin,wmax)
        cons = (
                {'type': 'ineq', 'fun': lambda x: x - wmin}, \
                {'type': 'ineq', 'fun': lambda x: -x + wmax},\
                {'type': 'ineq', 'fun': lambda x: self.h_barrier-np.sum(2*self.delta_e_barrier*np.array([self.v_e*np.cos(x)-self.v_barrier,self.v_e*np.sin(x)]))/np.linalg.norm(self.delta_e_barrier)},\
                {'type': 'ineq', 'fun': lambda x: (self.h_pv-np.sum(self.delta_pv*np.array([self.v_e*np.cos(x)-self.v_p*np.cos(self.state_p[2]), self.v_e*np.sin(x)-self.v_p*np.sin(self.state_p[2])]))/np.linalg.norm(self.delta_pv))})
        return cons

    def composite_CBF(self,nominal_control):
        self.delta_e_barrier = np.array(self.state_e)-np.array(self.barrier[0:2])
        self.h_barrier = np.linalg.norm(self.delta_e_barrier) - self.barrier_judge
        self.delta_pv = np.array(self.state_e)-np.array(self.state_p[0:2])
        self.h_pv = np.linalg.norm(self.delta_pv) - self.pv_judge
        x0 = 0
        args = (-self.limie_phi, self.limie_phi)
        composite_cons = self.composite_cons(args)
        res = minimize(fun=self.QPfun( nominal_control[0]), x0=x0, constraints=composite_cons)
        action = res.x
        # print(res.success)
        return action


    def target_cons(self,args):
        wmin,wmax = args
        # print(wmin,wmax)
        cons = (
                {'type': 'ineq', 'fun': lambda x: x - wmin}, \
                {'type': 'ineq', 'fun': lambda x: -x + wmax},\
                {'type': 'ineq', 'fun': lambda x: self.h_x-np.sum(2*self.delta_p*np.array([self.v_e*np.cos(x),self.v_e*np.sin(x)]))})
        return cons

    def target_CBF(self, nominal_control):
        self.delta_p = np.array(self.state_e)-self.target_point
        self.h_x = self.target_judge - np.linalg.norm(self.delta_p)
        x0 = nominal_control[0]
        args = (-self.limie_phi,self.limie_phi)
        target_cons = self.target_cons(args)
        # print(np.linalg.norm(self.delta_p),self.h_x<0)
        if self.h_x<0:
            res = minimize(fun=self.QPfun(nominal_control[0]), x0=x0, constraints=target_cons)
            action =res.x
        else:
            action = self.phi

        res = minimize(fun=self.QPfun(nominal_control[0]), x0=x0, constraints=target_cons)
        action = res.x
        return action






