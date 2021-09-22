# !/usr/bin/python
# -*- coding: utf-8 -*-
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
from multi_discrete import MultiDiscrete
import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist
# Twist - 线速度角速度
# pose - 位置：position为实际位置，
# orientation为方位 --四元数？一个旋转的向量 + 一个旋转的角度。
# 其中x,y,z 代表的是向量的三维坐标，w代表的是角度
class DoubleEscape:
    def __init__(self):
        self.env_type = 'discrete'
        self.name = 'double_escape_discrete'
        # 初始化一个node，名字为self.name  anonymous=True，
        # 表示后面定义相同的node名字时候，按照序号进行排列
        # log_level设置日志等级
        rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        # env properties
        self.rate = rospy.Rate(1000)
        self.max_episode_steps = 1000
        self.obs_dim = 5  # 每个agent的observation的维度
        self.act_dim = 1  # action的维度(个数)
        # self.obs_dim = 6  # 每个agent的observation的维度
        # self.act_dim = 2  # action的维度(个数)
        # self.observation_space_shape = (2, 6) # {r1, r2, s}: x, y, x_d, y_d, th, th_d
        # self.action_space = np.array([[1.5, pi / 3], [1.5, -pi / 3], [-1.5, pi / 3], [-1.5, -pi / 3]])
        self.observation_space = []
        self.action_space = []
        self.action_space_shape = []
        self.discrete_action_space = False
        self.damping = 0.5
        self.dt = 0.1
        self.accel = 5
        self.u_range = 1
        self.delta_p=0
        self.delta_e=0
        self.theta=0
        # self.action_n = np.array([d, a])
        self.action_n = np.zeros([2,5])

        for i in range(2):
            # total_action_space = []
            # if self.discrete_action_space:
            #     u_action_space = spaces.Discrete(self.act_dim)  # action的维度
            # else:
            #     u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32)
            # total_action_space.append(u_action_space)
            # if len(total_action_space) > 1:
            #     if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
            #         act_space = MultiDiscrete([[0, (act_space.n - 1)] for act_space in total_action_space])
            #     else:
            #         act_space = spaces.Tuple(total_action_space)
            #     self.action_space.append(act_space)
            # else:
            #     self.action_space.append(total_action_space[0])
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

            self.action_space.append(spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32))
            self.observation_space.append(spaces.Box(low=np.array([-np.inf,-np.pi,-np.pi,-10,-10]), high=np.array([np.inf,np.pi,np.pi,10,10]), dtype=np.float32))
        # robot properties
        self.model_states = ModelStates()
        # self.obs = np.zeros(self.observation_space_shape)
        # self.prev_obs = np.zeros(self.observation_space_shape)
        self.obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        self.prev_obs = [[0 for i in range(self.obs_dim)], [0 for i in range(self.obs_dim)]]
        self.status = ['deactivated']*2  # ['deactivated','deactivated']
        self.world_name = rospy.get_param('/world_name')  # 获取参数
        self.exit_width = rospy.get_param('/exit_width')
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 指定服务名来调用服务
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # topic publisher
        self.cmd_vel0_pub = rospy.Publisher("/logger0/cmd_vel", Twist, queue_size=1)
        self.cmd_vel1_pub = rospy.Publisher("/logger1/cmd_vel", Twist, queue_size=1)
        # topic subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)



    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")  # 等待服务器连接
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def setModelState(self, model_state):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def reset(self, init_pose1=None, init_pose2=None):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        # rospy.logdebug("\nStart Environment Reset")
        # set init pose
        self.resetWorld()
        self.obs = self._set_pose(init_pose1, init_pose2)
        self.prev_obs = self.obs
        # rospy.logerr("\nEnvironment Reset!!!\n")
        return self.obs

    def step(self, action_n):
        """
        obs, rew, done, info = env.step(action_indices)
        """
        # rospy.logdebug("\nStart environment step")
        # print(action_n)
        actions = []
        done_n = []
        reward = []
        for i in range(2):
            actions.append(action_n[i])
        # update status
            reward, done = self._compute_reward()
            # self.prev_obs = self.obs.copy() # make sure this happened after reward computing
        # reward_n.append(reward)
            done_n.append(done)
        self.prev_obs = self.obs  # make sure this happened after reward computing
        info = self.status
        self._get_observation()
        self._take_action(actions)

        # rospy.logdebug("\nEnd environment step\n")
        # print(reward)
        # print(done_n)
        return self.prev_obs, reward, done_n, info

    def _set_pose(self, pose1 = None, pose2 = None):
        """
        Set double_logger with a random or given pose.
        """
        # rospy.logdebug("\nStart setting model pose")
        logger1 = ModelState()
        logger2 = ModelState()
        logger1.model_name = "logger0"
        logger2.model_name = "logger1"
        # logger_pose.reference_frame = "world"
        logger1.pose.position.z = 0.09 # 小车高度
        logger2.pose.position.z = 0.09
        if pose1 is None: # random pose
            x1 = random.uniform(-4, 4)
            y1 = random.uniform(-4, 4)
            th1 = random.uniform(-pi, pi)
            while any([
                np.abs(x1 + 2*np.sin(th1)) > 4.8,
                np.abs(y1 - 2*np.cos(th1)) > 4.8
            ]):
                th1 = random.uniform(-pi, pi)
            quat1 = tf.transformations.quaternion_from_euler(0, 0, th1) # 四元数
            rospy.logdebug("Set model1 pose @ {}".format((x1, y1, th1)))
        if pose2 is None:
            x2 = random.uniform(-4, 4)
            y2 = random.uniform(-4, 4)
            th2 = random.uniform(-pi, pi)
            while any([
                np.abs(x2 + 2*np.sin(th2)) > 4.8,
                np.abs(y2 - 2*np.cos(th2)) > 4.8
            ]):
                th2 = random.uniform(-pi, pi)
            quat2 = tf.transformations.quaternion_from_euler(0, 0, th2)
            rospy.logdebug("Set model2 pose @ {}".format((x2, y2, th2)))
        else: # inialize accordingly
            assert pose1.shape == (3,)
            assert pose1[0] <= 4.5
            assert pose1[1] <= 4.5
            assert -pi<= pose1[2] <= pi # theta within [-pi,pi]
            assert np.abs(pose1[0] + 2*np.sin(pose[2])) <= 4.8
            assert np.abs(pose1[1] - 2*np.cos(pose[2])) <= 4.8
            assert pose2.shape == (3,)
            assert pose2[0] <= 4.5
            assert pose2[1] <= 4.5
            assert -pi<= pose2[2]<= pi # theta within [-pi,pi]
            assert np.abs(pose2[0] + 2*np.sin(pose2[2])) <= 4.8
            assert np.abs(pose2[1] - 2*np.cos(pose2[2])) <= 4.8
            x1 = pose1[0]
            y1 = pose1[1]
            th1 = pose1[2]
            x2 = pose2[0]
            y2 = pose2[1]
            th2 = pose2[2]
            quat1 = tf.transformations.quaternion_from_euler(0, 0, th1)
            quat2 = tf.transformations.quaternion_from_euler(0, 0, th2)

        rospy.logdebug("Set model pose1 @ {}".format(logger1.pose))
        rospy.logdebug("Set model pose2 @ {}".format(logger2.pose))
        logger1.pose.position.x = x1
        logger1.pose.position.y = y1
        logger1.pose.orientation.z = quat1[2]
        logger1.pose.orientation.w = quat1[3]
        logger2.pose.position.x = x2
        logger2.pose.position.y = y2
        logger2.pose.orientation.z = quat2[2]
        logger2.pose.orientation.w = quat2[3]
        # set pose until on spot
        self.unpausePhysics()
        zero_vel = np.zeros((2, self.act_dim))
        self._take_action(zero_vel)
        self.setModelState(model_state=logger1)
        self.setModelState(model_state=logger2)
        self._take_action(zero_vel)
        self._get_observation()
        self.pausePhysics()
        # rospy.logdebug("\nEnd setting model pose")

        return self.obs

    def _get_dist(self):
        # _get_dist()函数中
        id_logger0 = self.model_states.name.index("logger0")  # 追
        id_logger1 = self.model_states.name.index("logger1")  # 逃
        logger_pose0 = self.model_states.pose[id_logger0]
        logger_twist0 = self.model_states.twist[id_logger0]
        logger_pose1 = self.model_states.pose[id_logger1]
        logger_twist1 = self.model_states.twist[id_logger1]
        # print('self.logger_twist0.linear.x', logger_twist0.linear.x)
        quat0 = [
            logger_pose0.orientation.x,
            logger_pose0.orientation.y,
            logger_pose0.orientation.z,
            logger_pose0.orientation.w
        ]
        quat1 = [
            logger_pose1.orientation.x,
            logger_pose1.orientation.y,
            logger_pose1.orientation.z,
            logger_pose1.orientation.w
        ]
        euler0 = tf.transformations.euler_from_quaternion(quat0)
        euler1 = tf.transformations.euler_from_quaternion(quat1)
        # self.obs[0][0] = logger_pose0.position.x
        # self.obs[0][1] = logger_pose0.position.y
        # self.obs[0][2] = logger_pose1.position.x -logger_pose0.position.x
        # self.obs[0][3] = logger_pose1.position.y -logger_pose0.position.y
        # self.obs[0][4] = logger_twist0.linear.x
        # self.obs[0][5] = logger_twist0.linear.y
        # self.obs[0][6] = logger_twist1.linear.x
        # self.obs[0][7] = logger_twist1.linear.y
        # self.obs[0][8] = euler0[2]
        # self.obs[0][9] = euler1[2]
        # self.obs[0][10] = logger_twist0.angular.z
        # self.obs[1][0] = logger_pose1.position.x
        # self.obs[1][1] = logger_pose1.position.y
        # self.obs[1][2] = logger_pose0.position.x -logger_pose1.position.x
        # self.obs[1][3] = logger_pose0.position.y -logger_pose1.position.y
        # self.obs[1][4] = logger_twist1.linear.x
        # self.obs[1][5] = logger_twist1.linear.y
        # self.obs[1][6] = logger_twist0.linear.x
        # self.obs[1][7] = logger_twist0.linear.y
        # self.obs[1][8] = euler1[2]
        # self.obs[1][9] = euler0[2]
        # self.obs[1][10] = logger_twist1.angular.z
        self.obs[0][0] = logger_pose1.position.x - logger_pose0.position.x
        self.obs[0][1] = logger_pose1.position.y - logger_pose0.position.y
        self.obs[0][2] = logger_twist0.linear.x
        self.obs[0][3] = logger_twist0.linear.y
        self.obs[0][4] = euler0[2]  #绕z轴旋转的角度
        self.obs[0][5] = logger_twist0.angular.z # 角速度
        self.obs[1][0] = logger_pose0.position.x - logger_pose1.position.x
        self.obs[1][1] = logger_pose0.position.y - logger_pose1.position.y
        self.obs[1][2] = logger_twist1.linear.x
        self.obs[1][3] = logger_twist1.linear.y
        self.obs[1][4] = euler1[2]
        self.obs[1][5] = logger_twist1.angular.z
        pos = [0, 0]
        # print("logger0",logger_pose0.position.x,logger_pose0.position.y)
        # print("logger1",logger_pose1.position.x,logger_pose1.position.y)
        # pos[0] = self.obs[0][0] - self.obs[1][0]
        # pos[1] = self.obs[0][1] - self.obs[1][1]
        pos[0] = self.obs[0][0]
        pos[1] = self.obs[0][1]
        self.delta_p = np.arctan(self.obs[0][0]/self.obs[0][1])-euler0[2]
        self.delta_e = np.arctan(self.obs[0][0]/self.obs[0][1])-euler1[2]
        dist = np.sqrt(np.sum(np.square(pos)))
        return dist
    def _get_dist2(self):
        # _get_dist()函数中
        id_logger0 = self.model_states.name.index("logger0")  # 追
        id_logger1 = self.model_states.name.index("logger1")  # 逃
        logger_pose0 = self.model_states.pose[id_logger0]
        logger_twist0 = self.model_states.twist[id_logger0]
        logger_pose1 = self.model_states.pose[id_logger1]
        logger_twist1 = self.model_states.twist[id_logger1]
        # print('self.logger_twist0.linear.x', logger_twist0.linear.x)
        quat0 = [
            logger_pose0.orientation.x,
            logger_pose0.orientation.y,
            logger_pose0.orientation.z,
            logger_pose0.orientation.w
        ]
        quat1 = [
            logger_pose1.orientation.x,
            logger_pose1.orientation.y,
            logger_pose1.orientation.z,
            logger_pose1.orientation.w
        ]
        euler0 = tf.transformations.euler_from_quaternion(quat0)
        euler1 = tf.transformations.euler_from_quaternion(quat1)
        delta_x= logger_pose1.position.x - logger_pose0.position.x
        delta_y= logger_pose1.position.y - logger_pose0.position.y
        pos = [0, 0]
        # print("logger0",logger_pose0.position.x,logger_pose0.position.y)
        # print("logger1",logger_pose1.position.x,logger_pose1.position.y)
        # pos[0] = self.obs[0][0] - self.obs[1][0]
        # pos[1] = self.obs[0][1] - self.obs[1][1]
        pos[0] = delta_x
        pos[1] = delta_y
        self.theta = np.arctan(pos[1]/pos[0])
        self.delta_p = self.theta - euler0[2]
        self.delta_e = self.theta- euler1[2]
        dist = np.sqrt(np.sum(np.square(pos)))
        self.obs[0][0]=dist
        self.obs[0][1]=self.theta
        self.obs[0][2]=euler0[2]
        self.obs[0][3]=delta_x #添加
        self.obs[0][4]=delta_y
        self.obs[1][0]=dist
        self.obs[1][1] = self.theta
        self.obs[1][2] = euler1[2]
        self.obs[1][3]=delta_x
        self.obs[1][4]=delta_y
        print('euler',self.obs[0][2])
        return dist
    def _get_observation(self):
        """
        Get observation of double_logger's state
        Args:
        Returns:
            obs: array([...pose+vel0...,pose+vell...pose+vel1...])
        """
        # rospy.logdebug("\nStart getting observation")
        dist = self._get_dist2()
        # dist = self._get_dist()
        if dist <= 0.51:
            self.status[0] = "catch it"
            self.status[1] = "trapped"
        else:
            self.status[0] = "gaming"
            self.status[1] = "gaming"
        # rospy.logdebug("\nEnd getting observation")

    def _take_action(self, actions):
        """
        Publish cmd_vel according to an action index
        Args:
            i_act: array([ia0, ia1])
        Returns:
        """
        # rospy.logdebug("\nStart Taking Action")
        cmd_vel0 = Twist()
        if self.discrete_action_space:
            cmd_vel0.linear.x = 1.3
            cmd_vel0.angular.z = 4 * (actions[0][0]-actions[0][1]) #policy_new2
            # cmd_vel0.angular.z += 4 * (actions[0][1]-actions[0][2]) #policy_new3

            # cmd_vel0.linear.x = 8*abs(actions[0][1]-actions[0][2])
            # cmd_vel0.angular.z = 8*(actions[0][3]-actions[0][4])
        else:

            cmd_vel0.linear.x = 1.3
            cmd_vel0.angular.z = 4 * actions[0][0]
            print('cmd_vel0.angular.z', cmd_vel0.angular.z)
        cmd_vel1 = Twist()
        if self.discrete_action_space:
            # # 指向机器人前方; cmd_vel1.linear.y 指向机器人左方，一般为0
            cmd_vel1.linear.x = 1.2
            cmd_vel1.angular.z = 8 * (actions[1][0]-actions[1][1])   # #policy_new2
            # cmd_vel1.angular.z += 8 * (actions[1][1]-actions[1][2])   # #policy_new3

            # cmd_vel1.linear.x = 7*abs(actions[1][1]-actions[1][2])
            # cmd_vel1.angular.z = 7*(actions[1][3]-actions[1][4])

        else:

            cmd_vel1.linear.x = 1.25
            cmd_vel1.angular.z = 12 * actions[1][0]
            print('cmd_vel1.angular.z',cmd_vel1.angular.z)

            # print(actions[1][1])
        self.unpausePhysics()
        for _ in range(50):
            self.cmd_vel0_pub.publish(cmd_vel0)
            self.cmd_vel1_pub.publish(cmd_vel1)
            self.rate.sleep()
        # rospy.logdebug("cmd_vel0: {} \ncmd_vel1: {}".format(cmd_vel0, cmd_vel1))
        self.pausePhysics()
        # rospy.logdebug("\nEnd Taking Action\n")

    def _compute_reward(self):
        """
        Compute reward and done based on current status
        Return:
            reward
            done
        """
        # rospy.logdebug("\nStart Computing Reward")
        dist = self._get_dist2()
        # dist = self._get_dist()  # 原
        reward, done = np.zeros(2), False
        if any([
            'trapped' in self.status,
            'catch it' in self.status
        ]):
            reward[0] = 1000
            reward[1] = -1000
            done = True
        # else:
        #     reward[0] += -dist#-abs(2*self.theta)
        #     reward[1] += dist#+abs(2*self.theta)
        # elif dist>0.55 and dist<3:
        #     reward[0] =-abs(2*self.theta)
        #     reward[1] =abs(2*self.theta)
        else:
            reward[0] -= dist
            reward[1] += dist
        # rospy.logdebug("\nEnd Computing Reward\n")

        return reward, done

    def _model_states_callback(self, data):
        self.model_states = data

if __name__ == "__main__":
    env = DoubleEscape()
    num_steps = env.max_episode_steps
    obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    for t in range(num_steps):
        a = [np.random.randint(0, 2, size=2) for i in range(2)]
        # print('sd', a)
        # gaz# a = [1, 3]
        o, r, d, i = env.step(a)
        st += 1
        rospy.loginfo("\n-\nepisode: {}, step: {} \nobs: {}, act: {}, reward: {}, done: {}, info: {}".format(ep+1, st, o, a, r, d, i))
        if any(d):
            ep += 1
            st = 0
            obs = env.reset()
