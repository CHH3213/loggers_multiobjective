# encoding: utf-8
#!/usr/bin/env python
"""
Doubld escape environment with discrete action space
"""
from __future__ import absolute_import, division, print_function

from gym import spaces, core
from gym import spaces
from gym.envs.registration import EnvSpec
import sys
import os
import math
import numpy as np
from numpy import pi
from numpy import random
import time
from math import *
import rospy
# import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
# from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
# from gazebo_msgs.msg import ModelState, ModelStates
import copy

def quart_to_rpy(x,y,z,w):
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return r,p,y
def euler_to_quaternion(roll, pitch, yaw):
    x=sin(pitch/2)*sin(yaw/2)*cos(roll/2)+cos(pitch/2)*cos(yaw/2)*sin(roll/2)
    y=sin(pitch/2)*cos(yaw/2)*cos(roll/2)+cos(pitch/2)*sin(yaw/2)*sin(roll/2)
    z=cos(pitch/2)*sin(yaw/2)*cos(roll/2)-sin(pitch/2)*cos(yaw/2)*sin(roll/2)
    w=cos(pitch/2)*cos(yaw/2)*cos(roll/2)-sin(pitch/2)*sin(yaw/2)*sin(roll/2)
    # import tf
    # (x, y, z, w) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    return x, y, z, w

class DoubleEscape(core.Env):
    def __init__(self):
        self.env_type = 'continuous'
        self.name = 'omni_diff_environment'
        rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        # env properties
        self.rate = rospy.Rate(1000)
        self.flag = True
        # map parameter
        self.ve = 0.787 # 0.787/4
        self.vp = 1.0 # 1/4
        self.v_barrier = 0.7
        self.p_l = 1
        self.p_b = 1
        self.theta_bs = 0
        self.init_angle = 0
        self.old_state = ["straight", "cb_T1"]
        self.number = 1
        self.angular_p = 0
        # state parameter
        self.fuhao = 1
        self.close_barrier = 0
        self.angle_amount = 0
        self.model_states = ModelStates()

        # position quaternion euler
        self.logger0_pos_x = 0.0
        self.logger0_pos_y = 0.0
        self.logger0_x = 0.0
        self.logger0_y = 0.0
        self.logger0_z = 0.0
        self.logger1_y = 0.0
        self.logger1_z = 0.0
        self.logger1_w = 1.0
        self.logger1_euler = [0, 0, 0]

        # relative position
        self.distance = 0.0
        self.theta = 0
        self.re_x = 0
        self.re_y = 0
        self.u_range = 2*np.pi
        self.obs_dim = 2  # 每个agent的observation的维度,全乡+差分+移动障碍：3+3+3,障碍物的为：位置+速度
        self.act_dim = 1  # action的维度(个数)
        self.obs = np.zeros([3,3])

        self.action_space=spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # robot properties

        self.status = ['deactivated']*2
        # 初始距离
        self.target = np.array([3,2])
        self.distance_pv =3
        self.distance_barrier = 3
        self.distance_goal =  np.sqrt(np.sum(np.square(np.array(self.obs[1][:2]) - self.target )))
        self.cmd_vel0_pub = rospy.Publisher("/logger0/cmd_vel", Twist, queue_size=1)
        self.cmd_vel1_pub = rospy.Publisher("/logger1/cmd_vel", Twist, queue_size=1)
        self.cmd_vel2_pub = rospy.Publisher("/barrier/cmd_vel", Twist, queue_size=1)

        # topic subscriber
        self.logger_0_pose = rospy.Subscriber('/logger0/odom', Odometry, self.logger_0_pose_callback)
        self.logger_1_pose = rospy.Subscriber('/logger1/odom', Odometry, self.logger_1_pose_callback)
        self.barrier_pose = rospy.Subscriber('/barrier/odom', Odometry, self.barrier_pose_callback)

    # multisenser subscriber callback


    def logger_0_pose_callback(self, pos):
        '''logger0 position--pursuer'''
        self.logger0_x = pos.pose.pose.orientation.x
        self.logger0_y = pos.pose.pose.orientation.y
        self.logger0_z = pos.pose.pose.orientation.z
        self.logger0_w = pos.pose.pose.orientation.w
        # initial position(0,0)
        self.logger0_pos_x = pos.pose.pose.position.x
        self.logger0_pos_y = pos.pose.pose.position.y
        self.logger0_linear_x = pos.twist.twist.linear.x
        self.logger0_linear_y = pos.twist.twist.linear.y
        self.logger0_angular_z = pos.twist.twist.angular.z
        #rospy.loginfo("logger0_w\n %f", pos.pose.pose.orientation.w)
        rospy.loginfo("logger0_x\n %f", pos.pose.pose.position.x)
        rospy.loginfo("logger0_y\n %f", pos.pose.pose.position.y)

    def logger_1_pose_callback(self, pos):
        '''
        logger1 position--evader'''
        self.logger1_x = pos.pose.pose.orientation.x
        self.logger1_y = pos.pose.pose.orientation.y
        self.logger1_z = pos.pose.pose.orientation.z
        self.logger1_w = pos.pose.pose.orientation.w
        # initial position(1,0)
        self.logger1_pos_x = pos.pose.pose.position.x +  1.0  # initial distance  between two loggers
        self.logger1_pos_y = pos.pose.pose.position.y
        self.logger1_linear_x = pos.twist.twist.linear.x
        self.logger1_linear_y = pos.twist.twist.linear.y
        self.logger1_angular_z = pos.twist.twist.angular.z
        #rospy.loginfo("logger1_w\n %f", pos.pose.pose.orientation.w)
        rospy.loginfo("logger1_x\n %f", pos.pose.pose.position.x)
        rospy.loginfo("logger1_y\n %f", pos.pose.pose.position.y)

    def barrier_pose_callback(self, pos):
        '''
        barrier position--moving obstacle'''
        self.barrier_x = pos.pose.pose.orientation.x
        self.barrier_y = pos.pose.pose.orientation.y
        self.barrier_z = pos.pose.pose.orientation.z
        self.barrier_w = pos.pose.pose.orientation.w
        # initial position(1,0)
        self.barrier_pos_x = pos.pose.pose.position.x + 1.0  # initial distance  between two loggers
        self.barrier_pos_y = pos.pose.pose.position.y
        self.barrier_linear_x = pos.twist.twist.linear.x
        self.barrier_linear_y = pos.twist.twist.linear.y
        self.barrier_angular_z = pos.twist.twist.angular.z
        #rospy.loginfo("logger1_w\n %f", pos.pose.pose.orientation.w)
        rospy.loginfo("barrier_x\n %f", pos.pose.pose.position.x)
        rospy.loginfo("barrier_y\n %f", pos.pose.pose.position.y)

    # def get_pose_gazebo(self):
    #     id_logger0 = self.model_states.name.index("logger0")  # 追
    #     id_logger1 = self.model_states.name.index("logger1")  # 逃
    #     id_barrier = self.model_states.name.index("barrier")  # 逃
    #     logger_pose0 = self.model_states.pose[id_logger0]
    #     logger_pose1 = self.model_states.pose[id_logger1]
    #     barrier_pose = self.model_states.pose[id_barrier]
    #     self.logger0_x = logger_pose0.orientation.x
    #     self.logger0_y = logger_pose0.orientation.y
    #     self.logger0_z = logger_pose0.orientation.z
    #     self.logger0_w = logger_pose0.orientation.w
    #     self.logger0_pos_x = logger_pose0.position.x
    #     self.logger0_pos_y = logger_pose0.position.y
    #     self.logger1_x = logger_pose1.orientation.x
    #     self.logger1_y = logger_pose1.orientation.y
    #     self.logger1_z = logger_pose1.orientation.z
    #     self.logger1_w = logger_pose1.orientation.w
    #     self.logger1_pos_x = logger_pose1.position.x
    #     self.logger1_pos_y = logger_pose1.position.y
    #     self.barrier_pos_x = barrier_pose.position.x
    #     self.barrier_pos_y = barrier_pose.position.y
    #     self.barrier_x = barrier_pose.orientation.x
    #     self.barrier_y = barrier_pose.orientation.y
    #     self.barrier_z = barrier_pose.orientation.z
    #     self.barrier_w = barrier_pose.orientation.w



    # establish the strategy map
    def calculate_state(self):
        self.old_state = copy.deepcopy(self.status)
        # print('oldstate',self.old_state)
        if self.distance <= self.p_l:
            self.status[0] = "catch it"
            self.status[1] = "trapped"
        pv = (self.ve / self.vp)
        pd = (self.p_b / self.p_l)
        yc = (self.p_l / pv)
        cos_bs = pv
        self.theta_bs = np.arccos(cos_bs)
        bs_x = (self.p_l * np.sin(self.theta_bs))
        bs_y = (self.p_l * np.cos(self.theta_bs))
        tao_c = (self.p_l / self.ve)
        tao_s = (self.p_b / (self.vp * np.tan(self.theta_bs)))
        # calculate if the barrier is closed
        if tao_c <= tao_s:
            self.close_barrier = 1
            #print("closed barrier!!!!!")
        else:
            self.close_barrier = 0
        # if barrier is closed
        if self.close_barrier == 1:
            k = ((yc-bs_y)/(-bs_x))
            # -----------------------------------------------------------------直行范围------------------------------------------------------#
            if ( abs(self.theta) < 0.2) and (abs(self.re_y) > yc):
            # if (abs(abs(self.logger0_euler[2]) - abs(self.logger1_euler[2])) < 0.1) and (abs(self.re_y) > (self.p_l / pv + 0.3)):
                # print('ppppppppppppppppppp')
                self.status[0] = "straight"
                self.status[1] = "cb_T1"
            # elif (self.status[0] == "straight") and (self.theta < 0.45):
            elif (self.status[0] == "straight") and (self.theta <self.theta_bs):
                self.status[0] = "straight"
                self.status[1] = "cb_T23"
            else:
                self.status[0] = "rotate"
                self.status[1] = "cb_T23"
            # elif (k * abs(self.re_x) + yc + 0.5 - abs(self.re_y)) < 0:
            #     self.status[0] = "rotate"
            #     self.status[1] = "cb_T23"
            # else:
            #     self.status[0] = "straight"
            #     self.status[1] = "cb_T23"
            #     #print("---------------------------------------------------------------------------")

    # reset at the start of the round
    def reset(self):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        # rospy.logdebug("\nStart Environment Reset")
        # set init pose

        state = self.get_place()

        # rospy.logerr("\nEnvironment Reset!!!\n")
        return state
    
    # get present place from multisensor
    def get_place(self):
        self.get_pose_gazebo()
        quat0 = [
            self.logger0_x,
            self.logger0_y,
            self.logger0_z,
            self.logger0_w
        ]
        quat1 = [
            self.logger1_x,
            self.logger1_y,
            self.logger1_z,
            self.logger1_w
        ]
        #barrier
        quat2 = [
            self.barrier_x,
            self.barrier_y,
            self.barrier_z,
            self.barrier_w
        ]
        # self.logger0_euler = tf.transformations.euler_from_quaternion(quat0)
        # self.logger1_euler = tf.transformations.euler_from_quaternion(quat1)
        self.logger0_euler = quart_to_rpy(quat0[0],quat0[1],quat0[2],quat0[3])
        self.logger1_euler = quart_to_rpy(quat1[0],quat1[1],quat1[2],quat1[3])
        self.barrier_euler = quart_to_rpy(quat2[0], quat2[1], quat2[2], quat2[3])

        delta_x = self.logger1_pos_x - self.logger0_pos_x
        delta_y = self.logger1_pos_y - self.logger0_pos_y
        pos = [0, 0]
        pos[0] = delta_x
        pos[1] = delta_y
        # #print('c',delta_x)
        dist = np.sqrt(np.sum(np.square(pos)))
        # position in reduced space
        self.re_x = (pos[0] * np.sin(self.logger0_euler[2]) - pos[1] * np.cos(self.logger0_euler[2]))
        self.re_y = (pos[0] * np.cos(self.logger0_euler[2]) + pos[1] * np.sin(self.logger0_euler[2]))
        # self.distance = np.sqrt(np.sum(np.square(pos)))
        self.distance = np.sqrt(np.square(self.re_x) + np.square(self.re_y))

        self.theta = np.arctan(self.re_x/self.re_y)
        # self.re_x = self.re_x[0]
        # self.re_y = self.re_y[0]
        # #print('a', self.re_x)
        # all the information needed to calculate strategy
        # self.obs[0][0] = dist
        # self.obs[0][1] = self.theta
        # self.obs[0][2] = self.logger0_euler[2]
        # self.obs[0][3] = self.re_x #添加
        # self.obs[0][4] = self.re_y
        # self.obs[1][0] = dist
        # self.obs[1][1] = self.theta
        # self.obs[1][2] = self.logger1_euler[2]
        # self.obs[1][3] = self.re_x
        # self.obs[1][4] = self.re_y

        self.obs[0][0] = self.logger0_pos_x
        self.obs[0][1] = self.logger0_pos_y
        self.obs[0][2] = self.logger0_euler[2]
        self.obs[1][0] = self.logger1_pos_x
        self.obs[1][1] = self.logger1_pos_y
        self.obs[1][2] = self.logger1_euler[2]
        self.obs[2][0] = self.barrier_pos_x
        self.obs[2][1] = self.barrier_pos_y
        self.obs[2][2] = self.barrier_euler[2]
        self.obs = np.array(self.obs)
        state = [self.obs[0][0], self.obs[0][1],self.obs[0][2], 
                 self.obs[1][0], self.obs[1][1], self.obs[1][2], 
                 self.obs[2][0], self.obs[2][1], self.obs[2][2]]
        # print(state)
        return state

    def get_reward_done(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        done = False
        r = 0
        self.distance_pv = np.sqrt(np.sum(np.square(np.array(self.obs[0][:2]) - np.array(self.obs[1][:2]))))
        self.distance_goal =  np.sqrt(np.sum(np.square(np.array(self.obs[1][:2]) - self.target )))
        self.distance_barrier = np.sqrt(np.sum(np.square(np.array(self.obs[1][:2]-self.obs[2][:2]))))
        # print('distance',distance_pv)
        if self.distance_pv <= 0.6:
            r -=500
            done =True
        # elif self.distance_pv>1:
        #     r += 0.01*self.distance_pv

        #############1##################
        # if distance_goal<=0.3:
        #     r +=1000
        #     done =True
        # elif distance > 1 and distance_goal > 0.3:
        #     r -=0.01*distance_goal
        # elif distance <=1 and distance_goal>0.3:
        #     r+=0.1*distance
        ##############################
        ##########2##########
        if self.distance_goal<=0.3:
            r +=1000
            done =True
        else:
            r -= 0.01*self.distance_goal
        ##########################
        # if self.obs[1][0] > 8 or self.obs[1][0] < -8 or self.obs[1][1] > 8 or self.obs[1][1] < -8:
        if self.distance_barrier<=0.6:
            r -=500
            done =True
        # print('done',done)
        return r, done

    def take_action(self):
        """
        Publish pursuer control
        Returns:cmd_vel
        """
        # rospy.logdebug("\nStart Taking Action")
        cmd_vel0 = Twist()

        if self.status[0] == "straight":
            if(self.re_y > 0):
                cmd_vel0.linear.x = self.vp
                cmd_vel0.angular.z = 0
            else:
                cmd_vel0.linear.x = -self.vp
                cmd_vel0.angular.z = 0
        if self.status[0] == "rotate" and (self.theta > 0):
            cmd_vel0.linear.x = 0
            cmd_vel0.angular.z = (self.vp/self.p_b)
        if self.status[0] == "rotate" and (self.theta < 0):
            cmd_vel0.linear.x = 0
            cmd_vel0.angular.z = (-self.vp/self.p_b)

        if self.status[0] == "catch it":
                    # cmd_vel0.linear.x = 0
                    # cmd_vel0.angular.z = 0
                if(self.re_y > 0):
                    cmd_vel0.linear.x = self.vp
                    cmd_vel0.angular.z = 0
                else:
                    cmd_vel0.linear.x = -self.vp
                    cmd_vel0.angular.z = 0

        # cmd_vel1 = Twist()
        # print('state',self.status)
        # print('oldstate',self.old_state)

        # if self.status[1] == "cb_T23":
        #     # print('ksksksksdfsfs')
        #     # if (self.old_state == "cb_T1"):
        #     #     self.angle_amount += 1
        #     # self.init_angle = (self.angle_amount * self.theta_bs)
        #     if (self.old_state[1] == "cb_T1"):
        #         print('hahdfsasffsdfgghghahaah')
        #         self.init_angle = (self.logger0_euler[2] - self.theta_bs)
        #         # 好像没转上角度？
        #         self.number = self.number + 1
        #     cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
        #     cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        # if self.status[1] == "cb_T1":
        #     cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
        #     cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        # if self.status[1] == "trapped":
        #     cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
        #     cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        # print('init_angle',self.init_angle)
        # print('x',cmd_vel1.linear.x)
        # print('y',cmd_vel1.linear.y)


        # rospy.logdebug("cmd_vel0: {} \ncmd_vel1: {}".format(cmd_vel0, cmd_vel1))
        # self.pausePhysics()
        # rospy.logdebug("\nEnd Taking Action\n")
        return cmd_vel0

    def take_e_action(self):
        '''
        it's the evader's optimal control
        '''
        cmd_vel1 = Twist()
        # print('state',self.status)
        # print('oldstate',self.old_state)

        if self.status[1] == "cb_T23":
            # print('ksksksksdfsfs')
            # if (self.old_state == "cb_T1"):
            #     self.angle_amount += 1
            # self.init_angle = (self.angle_amount * self.theta_bs)
            if (self.old_state[1] == "cb_T1"):
                # print('hahdfsasffsdfgghghahaah')
                if (self.theta > 0):
                    self.init_angle = (self.logger0_euler[2] - self.theta_bs)
                else:
                    self.init_angle = (self.logger0_euler[2] + self.theta_bs)
                    # self.init_angle = (self.logger0_euler[2] - self.theta_bs)

                # 好像没转上角度？
                self.number = self.number + 1
            cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
            cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        if self.status[1] == "cb_T1":
            cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
            cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        if self.status[1] == "trapped":
            cmd_vel1.linear.y = self.ve * np.sin(self.init_angle)
            cmd_vel1.linear.x = self.ve * np.cos(self.init_angle)
        # print('init_angle',self.init_angle)
        # print('x',cmd_vel1.linear.x)
        # print('y',cmd_vel1.linear.y)
        return self.init_angle
    
    def take_barrier_action(self):
        '''
        it's a nominal control to barrier
        evader碰到障碍物后朝反方向逃跑
        '''
        angle = self.init_angle+np.pi
        return angle

    def take_target_action(self):
        '''
        it's a nominal control to barrier
        evader朝着目标前进
        '''
        delta_xy = np.array(self.obs[1][:2]) - self.target
        if delta_xy[0] !=0:
            self.target2evader = np.arctan(delta_xy[1]/delta_xy[0])  # 目标点与evader之间的指向
        else:
            self.target2evader = np.pi/2
        angle = self.target2evader
        return angle
        

    # with algorithm
    def step(self, action_e):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        # rospy.logdebug("\nStart environment step")
        # #print(action)
        info = self.status
        
        self.calculate_state()  # to evader train
        cmd_vel0 = self.take_action()   # pursuer optimal control      
        cmd_vel1 = Twist()
        cmd_vel1.linear.y = self.ve * np.sin(action_e)
        cmd_vel1.linear.x = self.ve * np.cos(action_e)
        # print('vy', cmd_vel1.linear.y)
        # print('vx', cmd_vel1.linear.x)
        cmd_vel2 = Twist()
        cmd_vel2.linear.x = self.v_barrier
        for _ in range(1):
            # self.cmd_vel0_pub.publish(cmd_vel0)  # pursuer
            rospy.sleep(0.0001)
            self.cmd_vel1_pub.publish(cmd_vel1)  # evader
            # self.cmd_vel2_pub.publish(cmd_vel2)  #obstacle
        # self.flag=False
        # update status
        reward, done = self.get_reward_done()
        state = self.get_place()

        self.rate.sleep()
        # rospy.logdebug("\nEnd environment step\n")
        # self.angular_p = cmd_vel0.angular.z
        return state, reward, done, info

if __name__ =='__main__':
    print(int(3e5))
