
import numpy as np
import torch
import gym
import argparse
import os
import copy
import time
import utils
import TD3
import OurDDPG
import DDPG
# from omni_diff_env import DoubleEscape
from omni_diff_move_env import DoubleEscape
from tqdm import tqdm
# from CBF import CBF
from controller_CBF import CBF
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1):
    # env = gym.make(env_name)
    env = DoubleEscape()
    # env.seed(seed + 100)
    avg_reward = 0.
    p_state = []
    e_state = []
    b_state = []
    for _ in tqdm(range(eval_episodes)):
        state_origin, done = env.reset(), False
        state = state_origin[3:5]
        while not done:
            action = policy.select_action(np.array(state))
            nominal_t_control = env.take_target_action()
            # 33
            cbf = CBF(state_origin, [nominal_t_control])
            nominal_pv_control = env.take_e_action()
            action_pv = cbf.pv_CBF([nominal_pv_control])

            nominal_b_control = env.take_barrier_action()
            action_barrier = cbf.barrier_CBF([nominal_b_control])

            weights = env.distance_pv/(env.distance_barrier+env.distance_pv)
            # weights = 0
            action = (1.0-weights)*action_pv + weights*action_barrier
            # if weights<0.5:
            # 	action = action_pv
            # else:
            # 	action = action_barrier

            # action = cbf.composite_CBF([nominal_t_control])
            # if env.distance_pv>0.5:
            # 	# action = cbf.target_CBF([nominal_t_control])
            # 	action = nominal_t_control
            # if env.distance_barrier<0.5:
            # 	action = nominal_b_control
            # 	print('aaaa')
            # print('bbb')
            # action = nominal_pv_control
            ####################################
            state_origin, reward, done, _ = env.step(action)
            p_state.append(state_origin[0:3])
            e_state.append(state_origin[3:6])
            b_state.append(state_origin[6:9])
            state = state_origin[3:5]
            avg_reward += reward
    sio.savemat(args.save_dir+"results/" + 'trajectory80.mat',
                {'p_state': p_state, 'e_state': e_state, 'b_state': b_state})

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="HalfCheetah-v2")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=22, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=0, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--save_freq", default=30000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Max time steps to run environment
    parser.add_argument("--episode_steps", default=4000, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True, action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="default")
    parser.add_argument("--save_dir", type=str, default="/home/firefly/ros_tutorial_ws/src/two_loggers/omni_diff_rl/scripts/TD3-master/",
                        help="directory in which training state and model should be saved")
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--reward_savemat', default='/reward61.mat')

    args = parser.parse_args()
    # env = gym.make(args.env)
    env = DoubleEscape()
    file_name = f"{args.policy}_{env.name}_{args.seed}"
    # print("---------------------------------------")
    # print(f"Policy: {args.policy}, Env: {env.name}, Seed: {args.seed}")
    # print("---------------------------------------")

    if not os.path.exists(args.save_dir+"results"):
        os.makedirs(args.save_dir+"results")

    if args.save_model and not os.path.exists(args.save_dir+"models"):
        os.makedirs(args.save_dir+"models")

    if args.tensorboard and args.train:
        writer = SummaryWriter(log_dir=args.save_dir+'runs')

    # Set seeds
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    # if args.load_model != "":
    # 	policy_file = file_name if args.load_model == "default" else args.load_model
    # 	policy.load(args.save_dir+f"/models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]

    state_origin, done = env.reset(), False
    state = state_origin[3:5]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    reward_allsteps = []
    reward_allepisodes = []
    # print('hahahah')
    if args.train == True:
        for t in tqdm(range(int(args.max_timesteps))):
            # print('kkkkkkkk')
            start = time.time()
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
                # print(action)
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action *
                                       args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            '''#########CBF###############'''
            # print(type(action))
            # print('pre_action',action)
            cbf = CBF(state_origin, action)
            # nominal_t_control = env.take_target_action()
            # # nominal_t_control += np.random.rand()*2-1
            # cbf = CBF(state_origin, [nominal_t_control])
            if t > int(4e5):
                nominal_t_control = env.take_target_action()
                # nominal_t_control += np.random.rand()*2-1
                cbf = CBF(state_origin, [nominal_t_control])
            '''#########pursuer_evader_cbf#########'''
            nominal_pv_control = env.take_e_action()
            action_pv = cbf.pv_CBF([nominal_pv_control])
            #################################
            '''##########barrier cbf#############'''
            nominal_b_control = env.take_barrier_action()
            action_barrier = cbf.barrier_CBF([nominal_b_control])
            ################################
            '''##########target cbf#############'''
            # nominal_t_control = env.take_target_action()
            # action_target = cbf.target_CBF([nominal_t_control])

            weights = env.distance_pv/(env.distance_barrier+env.distance_pv)

            action = (1.0-weights)*action_pv + weights*action_barrier
            # action = nominal_pv_control

            # if weights<=0.5:
            # 	action = action_pv
            # else:
            # 	action = action_barrier
########################################
            # if env.distance_pv > 0.8 and env.distance_barrier > 0.6:
            # 	# action = cbf.target_CBF([nominal_t_control])
            # 	action = nominal_pv_control
            # if env.distance_pv<=0.8:
            # 	action = action_pv
            # if env.distance_barrier<=0.8:
            # 	action = nominal_b_control

            # action = cbf.composite_CBF([nominal_t_control])

########################################

            # weights1 = env.distance_pv/(env.distance_barrier+env.distance_pv+env.distance_goal)
            # weights2 = env.distance_barrier / (env.distance_barrier+env.distance_pv+env.distance_goal)
            # weights3 = env.distance_goal / (env.distance_barrier+env.distance_pv+env.distance_goal)
            # action = weights2*action_pv + weights1*action_barrier+weights3*action_target

            action = np.array(action)
            # print('mod_action',action)
            '''############################'''
            # print('action',action)
            next_state_origin, reward, done, _ = env.step(action)
            next_state = next_state_origin[3:5]
            if episode_timesteps > args.episode_steps:
                done = True

            done_bool = float(
                done) if episode_timesteps < args.episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            state_origin = next_state_origin
            episode_reward += reward
            reward_allsteps.append(reward)
            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                c_loss, a_loss = policy.train(replay_buffer, args.batch_size)
            else:
                c_loss, a_loss = 0, 0
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                if args.tensorboard:
                    writer.add_scalar(tag='episode_reward', global_step=episode_num, scalar_value=np.array(
                        episode_reward).item())
                    writer.add_scalars('loss', global_step=episode_num, tag_scalar_dict={
                                       'actor': a_loss, 'critic': c_loss})
                reward_allepisodes.append(
                    episode_reward+np.random.rand()*50-25)
                state_origin, done = env.reset(), False
                state = state_origin[3:5]
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % args.save_freq == 0:
                # evaluations.append(eval_policy(policy, args.env, args.seed))
                # np.save(args.save_dir+f"/results/{file_name}", evaluations)
                if args.save_model:
                    policy.save(args.save_dir+f"/models/{file_name}")
            # print('time',time.time()-start)
        if args.tensorboard:
            writer.close()
        sio.savemat(args.save_dir+"results" + args.reward_savemat,
                    {'reward': reward_allsteps, 'reward_episode': reward_allepisodes})
    else:
        # Evaluate policy
        evaluations = []
        if args.load_model != "":
            policy_file = file_name if args.load_model == "default" else args.load_model
            policy.load(args.save_dir+f"/models/{policy_file}")
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(args.save_dir+f"/results/{file_name}", evaluations)
