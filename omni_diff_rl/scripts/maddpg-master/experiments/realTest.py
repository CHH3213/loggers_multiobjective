# !/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
# import tensorflow as tf
import time
import os
from sys import path
import tf_util as U
from maddpg import MADDPGAgentTrainer
# from maddpg import MADDPGEnsembleAgentTrainer
# import tensorflow.contrib.layers as layers
# from tf_slim import layers

# from tf_slim import rnn
import scipy.io as sio
import rospy
import cbf

path.append("./home/caohuanhui/catkin_ws/src/two_loggers/loggers_control/scripts")
# from my_double_ind import DoubleEscape
from pursuit_double_ind_copy import DoubleEscape


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=400, help="maximum episode length")  #每个episode的步数为400步
    #episodes的回数，先前默认60000,现在改成5000
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    #这里切换ddpg和maddpg
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../tmp/policy_new4/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="/home/caohuanhui/catkin_ws/src/two_loggers/loggers_control/scripts/maddpg-master/tmp/im_4/", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    # default=True出现图形界面
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()

# def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
#     # This model takes as input an observation and returns values of all actions
#     with tf.variable_scope(scope, reuse=reuse):
#         out = input
#         # out = tf.nn.dropout(out, 0.8)  # dropout
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
#         # out = tf.nn.dropout(out, 0.6)  # dropout
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
#         out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.tanh)
#         # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
#         return out
#
#
#
# def mlp_model_q(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
#     # This model takes as input an observation and returns values of all actions
#     with tf.variable_scope(scope, reuse=reuse):
#         out = input
#         # out = tf.nn.dropout(out, 0.8)  # dropout
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
#         # out = tf.nn.dropout(out, 0.8)  # dropout
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
#         # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.relu)
#         out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
#         return out
#


# 修改
def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    '''
    从env中读取agent数量，再根据env中的动作空间action_space和obs_shape_n创建agent训练实例。
    make the List of trainers
    @Output: List of trainers 返回训练实例对象trainers
    '''
    trainers = []
    model1 = mlp_model
    model2 = mlp_model_q
    trainer = MADDPGAgentTrainer
    # trainer = MADDPGEnsembleAgentTrainer

    # 将 adversaries 添加到 trainers列表
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model1, model2, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg'))) #action_reservoir可能需要修改
    # 将 agents 添加到 trainers列表
    for i in range(num_adversaries, 2):
        trainers.append(trainer(
            "agent_%d" % i, model1, model2, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))

    return trainers



#核心部分
def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = DoubleEscape()
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(2)]
        num_adversaries = 1
        ###
        # trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print("===============================")
            print('Loading previous state...')
            print("===============================")
            filename = 'gazeboSimulink'
            arglist.load_dir= os.path.join( arglist.load_dir, filename)
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(2)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        # saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        abound = 0.8
        bbound = 1.6
        t_start = time.time()
        print('Starting iterations...')

        # chh ===============================
        # 数据保存
        episode_reward = []
        step_episode = []
        position_ = []
        status_ = []
        volocity = []
        step = [i for i in range(arglist.max_episode_len+1)]
        action_save = []
        pos = [0, 0, 0]
        file_folder_name = "/home/firefly/chh_ws/src/two_loggers/loggers_control/scripts/maddpg-master/save_data/训练数据/barrier_4/"  # policy_continueTest
        if not os.path.exists(file_folder_name):
            os.makedirs(file_folder_name)
        if_first = True
        step_count = 0
        while True:
            action_n = np.array([0,0])
            # action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # print('action_n',action_n)
            # pos[0] = obs_n[0][0]
            # pos[1] = obs_n[0][1]
            # pos[2] = obs_n[0][4]
            # ct = np.sqrt(np.square(pos[0]) + np.square((pos[1])))
            # ct = pos[0] / ct
            # theta = np.arccos(ct)
            # if pos[1] > 0:
            #     theta = theta
            # elif pos[1] <= 0:
            #     theta = -theta
            # pos[2] = theta - pos[2]
            # if pos[2] > 3.142:
            #     pos[2] -= 6.28
            # elif pos[2] < -3.142:
            #     pos[2] += 6.28
            # temp = pos[2]
            # if np.abs(pos[2]) > np.abs(8*(1.8 - np.sqrt(obs_n[0][2]**2 + obs_n[0][3]**2))):
            #     temp = np.abs(8*(1.8 - np.sqrt(obs_n[0][2]**2 + obs_n[0][3]**2)))
            # if pos[2] < 0:
            #     pos[2] = -2
            # if pos[2] > 0:
            #     pos[2] = 2
            # if pos[2] == 0:
            #     pos[2] = 0
            # action_n[0] = [1.5, -pos[2]]
            # action_n = cbf.cbf(obs_n, action_n)
            # action_n[1] = [0, 0]
            # 默认策略
            from pursuer_default import default_p
            from evader_default import default_e
            # action_n[0]=default_p(obs_n,action_n)
            # action_n[1]=default_e(obs_n,action_n)
            # action_n = cbf.cbf(obs_n, action_n)
            # print(action_n[0])
            # 调用环境执行n个智能体生成的动作，环境返回n个智能体的新状态、奖励值、智能体是否死亡、其他信息
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            # print('sd',done_n)
            # print(
            #     "\nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(obs_n, action_n, rew_n, done_n, new_obs_n))
            episode_step += 1
            # done = all(done_n)
            done = any(done_n)
            # collect experience
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            # 保存训练数据到经验回放单元
            # print(rew_n)
	        # terminal = False
            # for i, agent in enumerate(trainers):
            #     agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            # 更新状态
            obs_n = new_obs_n
            # chh注释
            for i, rew in enumerate(rew_n):  # 更新总奖励和每个agent的奖励
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            # ## chh添加
            # # 数据保存
            if env.status[1] =='cb_T23' and if_first == True:
                step_count = episode_step
                print(episode_step)
                if_first =False
            step_episode.append(rew_n[0]) # 保存追赶的reward
            volocity.append(obs_n[0][4:8])
            action_save.append(action_n)
            position_.append([env.logger1_pos_x,env.logger1_pos_y,env.logger0_pos_x,env.logger0_pos_y,env.re_x,env.re_y])
            status_.append([env.status])
            if done:
                '''
                network_vs_default
                default_vs_default
                default_vs_network
                network_vs_network
                # '''
                if done:
                    print('*'*20)
                    print('done:', episode_step)
                # # 将数据保存成mat文件
                sio.savemat(file_folder_name + '/barrier.mat',
                            {'step': step, 'position': position_, 'status': status_,'step_count':step_count})
                # print("--------------------start---------------------")
                # print("dist", self.distance)
                # print("1_x", self.logger1_pos_x)
                # print("1_y", self.logger1_pos_y)
                # print("0_x", self.logger0_pos_x)
                # print("0_y", self.logger0_pos_y)
                # print("euler_1", self.logger1_euler[2])
                # print("euler_0", self.logger0_euler[2])
                # print("re_x", self.re_x)
                # print("re_y", self.re_y)
                # print("line", (k * abs(self.re_x) + yc - abs(self.re_y)))
                # print("status", self.status)
                print("---------------------end----------------------")
                print('save !!!')
                break #保存完之后退出
                episode_reward.append(step_episode) #将400个step保存进列表中

                # 重置
                step_episode = [] # 将每个episode的reward列表清空
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                print('**************************************************************************')
                print(len(episode_rewards))
                print('**************************************************************************')

                # print("lenth of episode_rewards is :", len(episode_rewards))
                # print(f"finished no.{num_terminal} episode!") # chh 2020/10/20

            # increment global step counter
            train_step += 1



            # 更新所有trainers参数update all trainers, if not in display or benchmark mode




if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
