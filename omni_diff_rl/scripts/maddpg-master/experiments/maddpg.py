#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf
import tf_util as U

from distributions import make_pdtype

from replay_buffer import ReplayBuffer

class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        # 更新目标网络参数，算法伪码最后一行。
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)  # tf.group()用于创造一个操作，可以将传入参数的所有操作进行分组
    return U.function([], [], updates=[expression])

# policy network
def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    '''
     p_func ,q_func -->  train.py mlp_model()
     num_units隐藏节点数
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # act_space_n为environment,py中的action_space
        # print('act_pdtype_n',act_pdtype_n) # 连续状态下输出两高斯分布
        # act_pdtype_n [<maddpg.common.distributions.DiagGaussianPdType object at 0x7fa5f443dba8>, <maddpg.common.distributions.DiagGaussianPdType object at 0x7fa5f42be5c0>]
        # set up placeholders观察空间和行为空间
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # print('====act_ph_n===',act_ph_n)
        # 连续状态下[<tf.Tensor 'agent_1_1/action0:0' shape=(?, 2) dtype=float32>,
        # <tf.Tensor 'agent_1_1/action1:0' shape=(?, 2) dtype=float32>]
        # print('-------------------p_index', p_index)
        # p_index指的是agent index
        # p网络只能获取agent index自身的观察空间
        p_input = obs_ph_n[p_index]
        # print('======p_input',p_input)  #p_input Tensor("observation1_1:0", shape=(?, 4), dtype=float32)

        # 策略网络 p_func  -->  train.py mlp_model()
        # int(act_pdtype_n[p_index].param_shape()[0]) -->  策略网络的输出维度
        # p_func 即为神经网络mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None)
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        # p是策略网络的输出
        #将参数包装在分布中
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        # print('==============asdsad', obs_ph_n) # <tf.Tensor 'observation0_1:0' shape=(?, 0) dtype=float32>,
        # print('========asddfgrtad', type(act_input_n[0]))
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        # 如果local_q_func为真（即为DDPG）
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        # q网络
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)
        # actor网络的损失函数
        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        # print('bbbbb', [obs_ph_n[p_index]]) #  [<tf.Tensor 'observation0:0' shape=(?, 0) dtype=float32>]
        # print('tttttttttttttt', act_sample) # <tf.Tensor 'agent_0_1/Softmax:0' shape=(?, 5) dtype=float32>)

        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        # print('act', act)  #('act', <function <lambda> at 0x7f3d44129938>)
        p_values = U.function([obs_ph_n[p_index]], p)


        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # q网络的输入包括所有智能体观察的状态,所有智能体输出的动作。
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        # print("***********************")
        # print("=======================")
        # print("q_input1:", q_input)  # q_input1: Tensor("agent_3/concat:0", shape=(?, 82), dtype=float32)
        #
        # print("=======================")
        # print("***********************")

        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
            # with tf.Session() as sess:
            #     print("=======================")
            #     print("=======================")
            #     # print("q_input2:", q_input.eval())
            #     # print("q_input2:", sess.run(q_input))
            #     print("q_input2:", q_input)  # q_input2: Tensor("agent_3/concat_1:0", shape=(?, 19), dtype=float32)
            #     print("=======================")
            #     print("=======================")
        # q_func输出是一维Tensor
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target_ph 是目标网络的输出？？
        # target_q += rew + self.args.gamma * (1.0 - done) * target_q_next  #论文伪码 给y(j)赋值一行
        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        # critic网络的损失函数
        loss = q_loss # + 1e-3 * q_reg
        # 使用optimizer来降低loss，其中变量表在q_func_vars，保证每个变量的梯度到 grad_norm_clipping -- 梯度剪切
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])#最后会返回一个outputs
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model1, model2,  obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name    #name of the agent
        self.n = len(obs_shape_n) #智能体数量\
        self.agent_index = agent_index  #特定智能体的索引
        self.args = args
        obs_ph_n = []
        # print('==========obs_shape_n', obs_shape_n)  # [(8,), (8,)]
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            # print('=============', U.BatchInput(obs_shape_n[i], name="observation" + str(i)))  # <maddpg.common.tf_util.BatchInput object at 0x7ff6ec31eb38>

        # Create all the functions necessary to train the model
        # self.q_train为q_train()函数的返回值train
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model2,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,  # 梯度剪切 --- 防止梯度爆炸 --- 梯度超过该值,直接设定为该值
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            # p网络和q网络的神经网络相同，都是mlp_model,可以自己定制model
            p_func=model1,
            q_func=model2,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)  # 经验池大小
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    # 策略网络输出动作
    def action(self, obs):
        obs = np.array(obs)
        # print('aaaa', type(obs))
        # print('self.act(obs[None])[0] ', self.act(obs[None])[0] )
        return self.act(obs[None])[0]  # 在数组索引中，加入None就相当于在对应维度加一维
        # obs_n = [obs]
        # return self.act(obs_n)[0]  # 在数组索引中，加入None就相当于在对应维度加一维

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        '''
        Input:  agents -->    所有的trainers
        Output: loss   -->   [loss of q_train,
                              loss of p_train,
                              mean of target_q,
                              mean of reward,
                              mean of next target_q,
                              std of target_q]
        '''

        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        # collect replay sample from all agents
        obs_n = []  # 'n' 表示agent总数。(清除过去的状态。)
        obs_next_n = []
        act_n = []
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index
        for i in range(self.n):  # 获取所有智能体的信息
            # 从buffer中c采样获取  observation, action, rewards, next observation, done
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)#获取自身的信息

        # train q network[Critic network]
        num_sample = 1
        target_q = 0.0
        # caohuanhui
        # obs_n + act_n + [target_q] ==> q network -->
        # Input is all observation and all action and the target_q -->
        # Output is value.
        target_act_next_n = [self.p_debug['target_act'](obs_next_n[0])]

        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]   # 根据observation生成agent下一步的action
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))  # 根据observation以及action计算target_q_next
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # caohuanhui 注释
        # train p network[Actor network]
        # obs_n + act_n ==> Policy network -->
        # Input is all observation and all action -->
        # Output is action.

        # 在这里p_train 是p_train函数的返回值train，
        # train=U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        # obs_n 与 act_n都是列表
        # 列表前面加星号作用是将列表中所有元素解开成独立的参数
        # def add(a, b):
        #    return a+b
        # data = [4,3]
        # print add(*data)
        # >>> 7
        # 字典前面加两个星号，是将字典解开成独立的元素作为形参。
        p_loss = self.p_train(*(obs_n + act_n))

        # 更新目标p、q网络的参数
        self.p_update()  # p_network: make_update_exp
        self.q_update()  # q_network: make_update_exp

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]


class MADDPGEnsembleAgentTrainer(AgentTrainer):
    def __init__(self, name, model1, model2, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.counter = 0

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i) + "_ag" + str(agent_index)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model2,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,  # 梯度剪切 --- 防止梯度爆炸 --- 梯度超过该值,直接设定为该值
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            # p网络和q网络的神经网络相同，都是mlp_model,可以自己定制model
            p_func=model1,
            q_func=model2,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = [ReplayBuffer(1e6) for i in range(self.n)]
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n, terminal):
        # Store transition in the replay buffer.
        for i in range(self.n):
            self.replay_buffer[i].add(obs_n[i], act_n[i], rew_n[i], new_obs_n[i], float(done_n[i]))

    def preupdate(self):
        self.counter += 1

    def update(self, agents):
        # replay buffer is not large enough
        if len(self.replay_buffer[0]) < self.max_replay_buffer_len:
            return

        if not self.counter % 100 == 0:  # update every 100 updates
            return

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_buffer[0].make_index(self.args.batch_size)
        for i in range(self.n):  # replay_buffer[0] is self
            obs, act, rew, obs_next, done = self.replay_buffer[i].sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer[0].sample_index(index)
        # clip rewards
        # rew = np.clip(rew, -1.0, +1.0)
        # target_q = 0.0
        # train q network
        target_act_next_n = [self.p_debug['target_act'](obs_next_n[0])]
        ptr = 0
        for agent in agents:
            if agent is self: continue
            ptr += 1
            target_act_next_n.append(agent.p_debug['target_act'](obs_next_n[ptr]))
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next)]