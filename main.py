import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
# from dynamics import NNDynamicsModel,compute_normalization
from policyNN import NNDynamicsModel, compute_normalization
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew

from datetime import datetime
from utils import Logger, configure_log_dir
import utils
import time

import torch


Monitor = False




def sample(env,
		   controller,
		   num_paths=10,
		   horizon=1000,
		   cost_fn=cheetah_cost_fn,
		   render=False,
		   verbose=False,
		   save_video=False,
		   ignore_done=True,
		   MPC=False
		   ):
	"""
		Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
		and returns rollouts by running on the env.
		Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
	"""
	paths = []

	""" YOUR CODE HERE """
	for i_path in range(num_paths):
		obs = env.reset()
		observations, actions, next_observations, rewards = [], [], [], []
		path = {}
		print('The num of sampling rollout : ', i_path)
		# for t in range(horizon):
		done = False
		t = 0

		start = time.time()  # caculate run time --start time point

		while not done:
			t += 1
			if render:
				env.render()
			obs = obs.astype(np.float64).reshape((1, -1))
			observations.append(obs)
			if MPC is True:
				action = controller.get_action4(obs)  # it costs much time
			else:
				action = controller.get_action(obs)  # it costs much time
			action = action.astype(np.float64).reshape((1, -1))
			actions.append(action)
			obs, reward, done, _ = env.step(action)
			obs = obs.astype(np.float64).reshape((1, -1))
			next_observations.append(obs)
			if not isinstance(reward, float):
				reward = np.asscalar(reward)
			rewards.append(reward)

			if not ignore_done:
				if done:
					print("Episode finished after {} timesteps".format(t + 1))
					break
			else:
				if t >= horizon:
					break

		end = time.time()
		runtime1 = end - start

		rewards = np.array(rewards, dtype=np.float64)
		rewards = np.transpose(rewards.reshape((1, -1)))  # shape(1000,0 ) convert to (1000,1)
		observations = np.concatenate(observations)
		next_observations = np.concatenate(next_observations)
		actions = np.concatenate(actions)

		data_dim = rewards.shape[0]
		returns = np.zeros((data_dim, 1))
		for i in range(data_dim):
			if i == 0:
				returns[data_dim - 1 - i] = rewards[data_dim - 1 - i]
			else:
				returns[data_dim - 1 - i] = rewards[data_dim - 1 - i] + returns[data_dim - 1 - i + 1]

		cost = trajectory_cost_fn(cost_fn, observations, actions, next_observations)

		path['observations'] = observations
		path['next_observations'] = next_observations
		path['actions'] = actions
		path['rewards'] = rewards
		path['returns'] = returns
		path['cost'] = cost

		paths.append(path)

	return paths


# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
	return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])


def plot_comparison(env, dyn_model):
	"""
	Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions.
	"""
	""" YOUR CODE HERE """
	pass


def train(env,
		  cost_fn,
		  logdir=None,
		  render=False,
		  learning_rate=1e-3,
		  onpol_iters=10,
		  dynamics_iters=60,
		  batch_size=512,
		  num_paths_random=10,
		  num_paths_onpol=10,
		  num_simulated_paths=10000,
		  env_horizon=1000,
		  mpc_horizon=15,
		  n_layers=2,
		  size=500,
		  activation='relu',
		  output_activation=None
		  ):
	"""

	Arguments:

	onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run.

	dynamics_iters              Number of iterations of training for the dynamics model
	|_                          which happen per iteration of the aggregation loop.

	batch_size                  Batch size for dynamics training.

	num_paths_random            Number of paths/trajectories/rollouts generated
	|                           by a random agent. We use these to train our
	|_                          initial dynamics model.

	num_paths_onpol             Number of paths to collect at each iteration of
	|_                          aggregation, using the Model Predictive Control policy.

	num_simulated_paths         How many fictitious rollouts the MPC policy
	|                           should generate each time it is asked for an
	|_                          action.

	env_horizon                 Number of timesteps in each path.

	mpc_horizon                 The MPC policy generates actions by imagining
	|                           fictitious rollouts, and picking the first action
	|                           of the best fictitious rollout. This argument is
	|                           how many timesteps should be in each fictitious
	|_                          rollout.

	n_layers/size/activations   Neural network architecture arguments.

	"""
	logz.configure_output_dir(logdir)
	# ========================================================
	#
	# First, we need a lot of data generated by a random
	# agent, with which we'll begin to train our dynamics
	# model.
	""" YOUR CODE HERE """
	random_controller = RandomController(env)

	paths = sample(env, random_controller, num_paths=num_paths_random, horizon=env_horizon, ignore_done=True)  # 10

	# ========================================================
	#
	# The random data will be used to get statistics (mean
	# and std) for the observations, actions, and deltas
	# (where deltas are o_{t+1} - o_t). These will be used
	# for normalizing inputs and denormalizing outputs
	# from the dynamics network.
	#
	""" YOUR CODE HERE """
	# concatenate observations & actions to numpy data_rand_x
	# concatenate (next_observations -observations) to numpy data_rand_y
	for i in range(num_paths_random):
		if i == 0:
			data_rand_x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
			data_rand_y = paths[i]['next_observations'] - paths[i]['observations']
		else:
			x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
			data_rand_x = np.concatenate((data_rand_x, x), axis=0)
			y = paths[i]['next_observations'] - paths[i]['observations']
			data_rand_y = np.concatenate((data_rand_y, y), axis=0)

	# Initialize data set D to Drand
	data_x = data_rand_x
	data_y = data_rand_y



	# ========================================================
	#
	# Build dynamics model and MPC controllers.
	#

	# sess = tf.Session()

	# dyn_model = NNDynamicsModel(env=env,
	# 							n_layers=n_layers,
	# 							size=size,
	# 							activation=activation,
	# 							output_activation=output_activation,
	# 							batch_size=batch_size,
	# 							iterations=dynamics_iters,
	# 							learning_rate=learning_rate,
	# 							normalization=normalization
	# 							)
	dyn_model = NNDynamicsModel(env=env,
								hidden_size=(500, 500),
								activation=activation,  #'tanh'
								).cuda()

	mpc_controller = MPCcontroller(env=env,
								   dyn_model=dyn_model,
								   horizon=mpc_horizon,
								   cost_fn=cost_fn,
								   num_simulated_paths=num_simulated_paths,
								   )

	# ========================================================
	#
	# Tensorflow session building.
	#
	# sess.__enter__()
	# tf.global_variables_initializer().run()

	# ========================================================
	#
	# Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset.
	# Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
	#

	# make dirs output
	if not (os.path.exists(logdir)):
		os.makedirs(logdir)
	path = os.path.join(logdir, 'model')
	if not (os.path.exists(path)):
		os.makedirs(path)

	for itr in range(onpol_iters):
		""" YOUR CODE HERE """

		if itr != 0:
			dyn_model.load_state_dict(torch.load(path + '/net_params.pkl'))

		if (itr % 9) == 0 or itr == (onpol_iters-1):
			logger = Logger(logdir, csvname='log' + str(itr))
			data = np.concatenate((data_x, data_y), axis=1)
			logger.log_table2csv(data)

		dyn_model.fit(data_x, data_y, epoch_size=dynamics_iters, batch_size=batch_size)

		torch.save(dyn_model.state_dict(), path + '/net_params.pkl')  # save only the parameters
		torch.save(dyn_model, path + '/net.pkl')  # save entire net

		print('-------------Itr %d-------------' % itr)
		print('Start time:\n')
		print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

		start = time.time()  # caculate run time --start time point
		# sample
		if Monitor is True:
			monitor_path = os.path.join(logdir, 'monitor' + str(itr))
			env = wrappers.Monitor(env, monitor_path, force=True)

		paths = sample(env, mpc_controller, num_paths=num_paths_onpol, horizon=env_horizon, render=False,
					   ignore_done=False, MPC=True)

		end = time.time()
		runtime2 = end - start
		print('runtime = ', runtime2)

		print('End time:\n')
		print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

		# concatenate observations & actions to numpy data_rand_x
		# concatenate (next_observations -observations) to numpy data_rand_y
		for i in range(num_paths_onpol):
			if i == 0:
				data_rl_x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
				data_rl_y = paths[i]['next_observations'] - paths[i]['observations']
			else:
				x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
				data_rl_x = np.concatenate((data_rl_x, x), axis=0)
				y = paths[i]['next_observations'] - paths[i]['observations']
				data_rl_y = np.concatenate((data_rl_y, y), axis=0)

		# Aggregate data
		data_x = np.concatenate((data_x, data_rl_x), axis=0)
		data_y = np.concatenate((data_y, data_rl_y), axis=0)

		costs = np.zeros((num_paths_onpol, 1))
		returns = np.zeros((num_paths_onpol, 1))
		for i in range(num_paths_onpol):
			costs[i] = paths[i]['cost']
			returns[i] = paths[i]['returns'][0]

		# LOGGING
		# Statistics for performance of MPC policy using
		# our learned dynamics model
		logz.log_tabular('Iteration', itr)
		# In terms of cost function which your MPC controller uses to plan
		logz.log_tabular('AverageCost', np.mean(costs))
		logz.log_tabular('StdCost', np.std(costs))
		logz.log_tabular('MinimumCost', np.min(costs))
		logz.log_tabular('MaximumCost', np.max(costs))
		# In terms of true environment reward of your rolled out trajectory using the MPC controller
		logz.log_tabular('AverageReturn', np.mean(returns))
		logz.log_tabular('StdReturn', np.std(returns))
		logz.log_tabular('MinimumReturn', np.min(returns))
		logz.log_tabular('MaximumReturn', np.max(returns))

		logz.dump_tabular()


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')  #HalfCheetah-v2
	# Experiment meta-params
	parser.add_argument('--exp_name', type=str, default='mb_mpc')
	parser.add_argument('--seed', type=int, default=3)
	parser.add_argument('--render', action='store_true')
	# Training args
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
	parser.add_argument('--onpol_iters', '-n', type=int, default=1)  # Aggregation iters 10
	parser.add_argument('--dyn_iters', '-nd', type=int, default=60)  # epochs 50
	parser.add_argument('--batch_size', '-b', type=int, default=512)
	# Data collection
	parser.add_argument('--random_paths', '-r', type=int, default=700)  # random path nums 700
	parser.add_argument('--onpol_paths', '-d', type=int, default=10)  # mpc path nums   30
	parser.add_argument('--ep_len', '-ep', type=int, default=1000)  # 1000   path length  200 1000
	# Neural network architecture args
	parser.add_argument('--n_layers', '-l', type=int, default=2)
	parser.add_argument('--size', '-s', type=int, default=500)
	# MPC Controller
	parser.add_argument('--mpc_horizon', '-m', type=int, default=15)  # mpc simulation H  10
	parser.add_argument('--simulated_paths', '-sp', type=int, default=10000)  # mpc  candidate  K 100
	args = parser.parse_args()

	# Set seed
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	# Make data directory if it does not already exist

	# Make env
	if args.env_name is 'HalfCheetah-v2':
		env = HalfCheetahEnvNew()
		cost_fn = cheetah_cost_fn

	env_name = args.env_name  # HalfCheetah-v2  My3LineDirect-v1
	cost_fn = cheetah_cost_fn
	env = gym.make(env_name)
	# env.set_goals(45 * 3.14 / 180.0)  # 角度要换成弧度

	logdir = configure_log_dir(logname=env_name, txt='-train')
	utils.LOG_PATH = logdir

	train(env=env,
		  cost_fn=cost_fn,
		  logdir=logdir,
		  render=args.render,
		  learning_rate=args.learning_rate,
		  onpol_iters=args.onpol_iters,
		  dynamics_iters=args.dyn_iters,
		  batch_size=args.batch_size,
		  num_paths_random=args.random_paths,
		  num_paths_onpol=args.onpol_paths,
		  num_simulated_paths=args.simulated_paths,
		  env_horizon=args.ep_len,
		  mpc_horizon=args.mpc_horizon,
		  n_layers=args.n_layers,
		  size=args.size,
		  activation='tanh',
		  output_activation=None,
		  )


if __name__ == "__main__":
	main()

print('Finished!')
