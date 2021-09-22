# encoding: utf-8
#!/usr/bin/env python
"""
Doubld escape environment with discrete action space
"""
from __future__ import absolute_import, division, print_function

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import sys
import os
import math
import numpy as np
from numpy import pi
from numpy import random
import time

import rospy
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates
import copy
class DoubleEscape:

    def __init__(self):
        self.env_type = 'discrete'
        self.name = 'double_escape_discrete'
        rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        # env properties
        self.rate = rospy.Rate(1000)
        self.max_episode_steps = 1000
        self.observation_space_shape = (2, 6) # {r1, r2, s}: x, y, x_d, y_d, th, th_d
        self.discrete_action_space = False
        self.flag = True
        # map parameter
        self.ve = 0.9 # 0.787/4
        self.vp = 1.0 # 1/4
        self.p_l = 1.0
        self.p_b = 1.0
        self.theta_bs = 0
        self.init_angle = 0
        self.old_state = ["straight", "cb_T1"]
        self.number = 1

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
        self.logger0_w = 1.0
        self.logger0_euler = [0, 0, 0]
        self.logger1_pos_x = 0.0
        self.logger1_pos_y = 0.0
        self.logger1_x = 0.0
        self.logger1_y = 0.0
        self.logger1_z = 0.0
        self.logger1_w = 1.0
        self.logger1_euler = [0, 0, 0]

        # relative position
        self.distance = 0.0
        self.theta = 0
        self.re_x = 0
        self.re_y = 0
        self.count = 0
        self.u_range = 1
        self.obs_dim = 5  # 每个agent的observation的维度
        self.act_dim = 1  # action的维度(个数)
        self.observation_space = []
        self.action_space = []

        for i in range(2):
            self.action_space.append(spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32))
            self.observation_space.append(spaces.Box(low=np.array([-np.inf,-np.pi,-np.pi,-10,-10]), high=np.array([np.inf,np.pi,np.pi,10,10]), dtype=np.float32))

        # self.action_space_shape = ()
        self.action_reservoir = np.array([[1.5, pi/3], [1.5, -pi/3], [-1.5, pi/3], [-1.5, -pi/3]])
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # robot properties
        # self.obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        # self.prev_obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        self.obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        self.prev_obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        self.status = ['deactivated']*2

        # topic publisher
        self.cmd_vel0_pub = rospy.Publisher("/logger0/cmd_vel", Twist, queue_size=1)
        self.cmd_vel1_pub = rospy.Publisher("/logger1/cmd_vel", Twist, queue_size=1)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)
        # self.logger_0_pose = rospy.Subscriber('/logger0/odom', Odometry, self.logger_0_pose_callback)
        # self.logger_1_pose = rospy.Subscriber('/logger1/odom', Odometry, self.logger_1_pose_callback)

    # multisenser subscriber callback

    def setModelState(self, model_state):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def _model_states_callback(self, data):
        self.model_states = data

    def get_pose_gazebo(self):
        id_logger0 = self.model_states.name.index("logger0")  # 追
        id_logger1 = self.model_states.name.index("logger1")  # 逃
        logger_pose0 = self.model_states.pose[id_logger0]
        logger_pose1 = self.model_states.pose[id_logger1]
        self.logger0_x = logger_pose0.orientation.x
        self.logger0_y = logger_pose0.orientation.y
        self.logger0_z = logger_pose0.orientation.z
        self.logger0_w = logger_pose0.orientation.w
        self.logger0_pos_x = logger_pose0.position.x
        self.logger0_pos_y = logger_pose0.position.y
        self.logger1_x = logger_pose1.orientation.x
        self.logger1_y = logger_pose1.orientation.y
        self.logger1_z = logger_pose1.orientation.z
        self.logger1_w = logger_pose1.orientation.w
        self.logger1_pos_x = logger_pose1.position.x
        self.logger1_pos_y = logger_pose1.position.y

    # def logger_0_pose_callback(self, pos):
    #     self.logger0_x = pos.pose.pose.orientation.x
    #     self.logger0_y = pos.pose.pose.orientation.y
    #     self.logger0_z = pos.pose.pose.orientation.z
    #     self.logger0_w = pos.pose.pose.orientation.w
    #     self.logger0_pos_x = pos.pose.pose.position.x
    #     self.logger0_pos_y = pos.pose.pose.position.y
    #     # rospy.loginfo("logger0_x\n %f", pos.pose.pose.position.x)
    #     # rospy.loginfo("logger0_y\n %f", pos.pose.pose.position.y)

    # def logger_1_pose_callback(self, pos):
    #     self.logger1_x = pos.pose.pose.orientation.x
    #     self.logger1_y = pos.pose.pose.orientation.y
    #     self.logger1_z = pos.pose.pose.orientation.z
    #     self.logger1_w = pos.pose.pose.orientation.w
    #     self.logger1_pos_x = (pos.pose.pose.position.x + 1)
    #     self.logger1_pos_y = pos.pose.pose.position.y
    #     # rospy.loginfo("logger1_x\n %f", pos.pose.pose.position.x)
    #     # rospy.loginfo("logger1_y\n %f", pos.pose.pose.position.y)

    # establish the strategy map
    def calculate_state(self):
        # self.old_state = self.status
        self.get_place()
        self.old_state = copy.deepcopy(self.status)
        # print('oldstate',self.old_state)
        if self.distance <= (self.p_l - 0.1):
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
            #print("--------------------start---------------------")
            #print("dist", self.distance)
            #print("1_x", self.logger1_pos_x)
            #print("1_y", self.logger1_pos_y)
            #print("0_x", self.logger0_pos_x)
            #print("0_y", self.logger0_pos_y)
            # print("euler_1", self.logger1_euler[2])
            # print("euler_0", self.logger0_euler[2])
            #print("euler_v2", self.theta)
            #print("re_x", self.re_x)
            #print("re_y", self.re_y)
            #print("line", (k * abs(self.re_x) + yc - abs(self.re_y)))
            #print("status", self.status)
            #print("---------------------end----------------------")
            # #print(self.re_x)
            # #print(self.re_y)
            # if (abs(self.re_x) <= 0.3) and (abs(self.re_y) > (self.p_l/pv)):
            if (abs(abs(self.logger0_euler[2]) - abs(self.logger1_euler[2])) < 0.1) and (abs(self.re_y) > (self.p_l / pv + 0.3)):
                # print('ppppppppppppppppppp')
                self.status[0] = "straight"
                self.status[1] = "cb_T1"
            elif (self.status[0] == "straight") and (self.theta < 0.45):
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
        logger1 = ModelState()
        logger2 = ModelState()
        logger1.model_name = "logger0"
        logger2.model_name = "logger1"
        # logger_pose.reference_frame = "world"
        logger1.pose.position.z = 0.09  # 小车高度
        logger2.pose.position.z = 0.09
        logger1.pose.position.x = 0
        logger1.pose.position.y = 0
        quart1 = tf.transformations.quaternion_from_euler(0, 0, 0)
        logger1.pose.orientation.x = quart1[0]
        logger1.pose.orientation.y = quart1[1]
        logger1.pose.orientation.z = quart1[2]
        logger1.pose.orientation.w = quart1[3]
        logger2.pose.position.x = 3
        logger2.pose.position.y = 0
        quart2 = tf.transformations.quaternion_from_euler(0, 0, 0)
        logger2.pose.orientation.x = quart1[0]
        logger2.pose.orientation.y = quart1[1]
        logger2.pose.orientation.z = quart1[2]
        logger2.pose.orientation.w = quart1[3]
        self.setModelState(model_state=logger1)
        self.setModelState(model_state=logger2)
        self.get_place()
        self.prev_obs = self.obs
        # self.y = np.array([obs[0,1], obs[1,1]])
        # self.prev_y = self.y.copy()
        # rospy.logerr("\nEnvironment Reset!!!\n")
        return self.obs
    
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
        self.logger0_euler = tf.transformations.euler_from_quaternion(quat0)
        self.logger1_euler = tf.transformations.euler_from_quaternion(quat1)
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
        self.distance = np.sqrt(np.sum(np.square(pos)))
        self.theta = np.arctan((-self.re_x)/self.re_y)
        # self.re_x = self.re_x[0]
        # self.re_y = self.re_y[0]
        # #print('a', self.re_x)
        # all the information needed to calculate strategy
        self.obs[0][0] = dist
        self.obs[0][1] = self.theta
        self.obs[0][2] = self.logger0_euler[2]
        self.obs[0][3] = self.re_x #添加
        self.obs[0][4] = self.re_y
        self.obs[1][0] = dist
        self.obs[1][1] = self.theta
        self.obs[1][2] = self.logger1_euler[2]
        self.obs[1][3] = self.re_x
        self.obs[1][4] = self.re_y
        return dist

    def compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        self.count += 1
        # rospy.logdebug("\nStart Computing Reward")
        reward, done = np.zeros(2), False
        if (self.count >= 50000):
            done = True
        # rospy.logdebug("\nEnd Computing Reward\n")
        return reward, done

    # take next action: publish to control the model
    def take_action(self, actions):
        """
        Publish cmd_vel according to an action index
        Args:
            i_act: array([ia0, ia1])
        Returns:
        """
        # rospy.logdebug("\nStart Taking Action")
        cmd_vel0 = Twist()
        if self.discrete_action_space:
            cmd_vel0.linear.x = 0
            cmd_vel0.angular.z = 0
        else:
            if self.status[0] == "straight":
                cmd_vel0.linear.x = self.vp
                cmd_vel0.angular.z = 0
            if self.status[0] == "rotate":
                cmd_vel0.linear.x = 0
                cmd_vel0.angular.z = (self.vp/self.p_b)
            if self.status[0] == "catch it":
                cmd_vel0.linear.x = 0
                cmd_vel0.angular.z = 0

        cmd_vel1 = Twist()
        print('state',self.status)
        print('oldstate',self.old_state)
        if self.discrete_action_space:
            # # 指向机器人前方; cmd_vel1.linear.y 指向机器人左方，一般为0
            cmd_vel1.linear.x = 0
            cmd_vel1.angular.z = 0
        else:
            if self.status[1] == "cb_T23":
                # print('ksksksksdfsfs')
                # if (self.old_state == "cb_T1"):
                #     self.angle_amount += 1
                # self.init_angle = (self.angle_amount * self.theta_bs)
                if (self.old_state[1] == "cb_T1"):
                    print('hahdfsasffsdfgghghahaah')
                    self.init_angle = (self.logger0_euler[2] - self.theta_bs)
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
        for _ in range(10):
            self.cmd_vel0_pub.publish(cmd_vel0)
            # rospy.sleep(0.00001)
            #
            # # cmd_vel1.angular.z = 0
            # cmd_vel1.linear.x = 0
            # cmd_vel1.linear.y = 1
            self.cmd_vel1_pub.publish(cmd_vel1)
            # self.flag=False
            self.rate.sleep()

        # self.old_state = self.status

        # rospy.logdebug("cmd_vel0: {} \ncmd_vel1: {}".format(cmd_vel0, cmd_vel1))
        # self.pausePhysics()
        # rospy.logdebug("\nEnd Taking Action\n")
    
    # with algorithm
    def step(self, action_n):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        # rospy.logdebug("\nStart environment step")
        # #print(action)

        actions = []
        done_n = []
        reward = []

        for i in range(2):
            actions.append(action_n[i])
            # update status
            reward, done = self.compute_reward()
            # self.prev_obs = self.obs.copy() # make sure this happened after reward computing
            # reward_n.append(reward)
            done_n.append(done)
        self.prev_obs = self.obs  # make sure this happened after reward computing
        info = self.status
        print()
        self.calculate_state()
        self.take_action(actions)
        print()

        # rospy.logdebug("\nEnd environment step\n")
        # #print(reward)
        # #print(done_n)
        return self.prev_obs, reward, done_n, info

if __name__ == "__main__":
    env = DoubleEscape()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    for t in range(num_steps):
        a = [np.random.randint(0, 2, size=2) for i in range(2)]
        # #print('sd', a)
        # gaz# a = [1, 3]
        o, r, d, i = env.step(a)
        st += 1
        # rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if any(d):
            ep += 1
            st = 0
            obs = env.reset()