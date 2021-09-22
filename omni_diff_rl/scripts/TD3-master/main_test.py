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
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	# eval_env = DoubleEscape()

	# eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in tqdm(range(eval_episodes)):
		env.render()
		state, done = eval_env.reset(), False

		while not done:
			env.render()
			action = policy.select_action(np.array(state))
			############################33
			# cbf = CBF(state, action)
			# nominal_pv_control = env.take_e_action()
			# action_pv = cbf.pv_CBF([nominal_pv_control])

			# nominal_b_control = env.take_barrier_action()
			# action_barrier = cbf.barrier_CBF([nominal_b_control])

			# weights = env.distance_pv/(env.distance_barrier+env.distance_pv)
			# action = (1.0-weights)*action_pv + weights*action_barrier
			####################################
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--policy", default="TD3")
	parser.add_argument("--env", default="Pendulum-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=666, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used
	parser.add_argument("--save_freq", default=10000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
	parser.add_argument("--episode_steps", default=200, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--save_dir", type=str, default="/home/firefly/ros_tutorial_ws/src/two_loggers/omni_diff_rl/scripts/TD3-master/", help="directory in which training state and model should be saved")
	parser.add_argument('--train', default=True, action="store_true")
	parser.add_argument('--tensorboard', default=True, action="store_true")
	parser.add_argument('--reward_savemat', default='/reward4.mat')

	args = parser.parse_args()
	env = gym.make(args.env)
	# env = DoubleEscape()
	# file_name = f"{args.policy}_{env.name}_{args.seed}"
	file_name = f"{args.policy}_{'Pendulum'}_{args.seed}"
	# print("---------------------------------------")
	# print(f"Policy: {args.policy}, Env: {env.name}, Seed: {args.seed}")
	# print("---------------------------------------")
	
	if not os.path.exists(args.save_dir+"results"):
		os.makedirs(args.save_dir+"results")

	if args.save_model and not os.path.exists(args.save_dir+"models"):
		os.makedirs(args.save_dir+"models")

	if args.tensorboard and args.train:
		writer =  SummaryWriter(log_dir=args.save_dir+'runs' )

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

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	reward_allsteps = []
	# print('hahahah')
	if args.train==True:
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
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

			# Perform action
			'''#########CBF###############'''
			# print(type(action))
			# print('pre_action',action)
			# cbf = CBF(state,action)
			
			# nominal_pv_control = env.take_e_action()
			# action_pv = cbf.pv_CBF([nominal_pv_control])

			# nominal_b_control = env.take_barrier_action()
			# action_barrier = cbf.barrier_CBF([nominal_b_control])
			
			# weights = env.distance_pv/(env.distance_barrier+env.distance_pv)
			# action = (1.0-weights)*action_pv +weights*action_barrier
			# action = np.array(action)
			# print('mod_action',action)
			'''############################'''
			# print('action',action)
			next_state, reward, done, _ = env.step(action) 
			if episode_timesteps > args.episode_steps:
				done =True

			done_bool = float(done) if episode_timesteps <  args.episode_steps else 0


			# Store data in replay buffer
			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward
			reward_allsteps.append(reward)
			# Train agent after collecting sufficient data
			if t >= args.start_timesteps:
				a_loss,c_loss = policy.train(replay_buffer, args.batch_size)
			else:
				a_loss,c_loss = 0,0
			if done: 
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset environment
				if args.tensorboard:
					writer.add_scalar(tag='episode_reward',global_step=episode_num, scalar_value=np.array(episode_reward).item())
					writer.add_scalars('loss', global_step=episode_num, tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

				state, done = env.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1 

			# Evaluate episode
			if (t + 1) % args.save_freq == 0:
				# evaluations.append(eval_policy(policy, args.env, args.seed))
				# np.save(args.save_dir+f"/results/{file_name}", evaluations)
				if args.save_model: policy.save(args.save_dir+f"/models/{file_name}")
			# print('time',time.time()-start)
		if args.tensorboard:
			writer.close()
		sio.savemat(args.save_dir+"results" + args.reward_savemat, {'reward':reward_allsteps})
	else:
		# Evaluate policy
		evaluations = []
		if args.load_model != "":
			policy_file = file_name if args.load_model == "default" else args.load_model
			policy.load(args.save_dir+f"/models/{policy_file}")
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(args.save_dir+f"/results/{file_name}", evaluations)

