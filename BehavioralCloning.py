import csv
import numpy as np
from policy import NNPolicyModel
from policyNN import Policy
import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from dynamics import NNDynamicsModel,compute_normalization
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew

from datetime import datetime
from utils import Logger,configure_log_dir
import utils
import time



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
		 activation=tf.nn.relu,
		 output_activation=None
		 ):
	##-----------------------------------------------------------------------------
	## get data from teacher's behavior

	# 初始化data
	state, action = [], []
	for i in range(27100):  ##the number of row!
		state.append([])
		action.append([])

	with open('log9.csv', 'r') as f:
		f_csv = csv.reader(f)
		n_row = 0
		for row in f_csv:
			if n_row != 0:  # 第0行沒有資訊
				for i in range(0, 20):
					state[n_row - 1].append(float(row[i]))
				for i in range(20, 26):
					action[n_row - 1].append(float(row[i]))
			n_row = n_row + 1

	state = np.array(state)
	action = np.array(action)
	# print(state[0])
	# print(action[0])

	## train model to match expert traject.
	scaler_x = 0
	scaler_y = 0
	normalization = [scaler_x, scaler_y]
	## build NN model to train
	NNPolicy = NNPolicyModel(env=env,
							  n_layers=n_layers,
							  size=size,
							  activation=activation,
							  output_activation=output_activation,
							  batch_size=batch_size,
							  iterations=dynamics_iters,
							  learning_rate=learning_rate,
							  normalization=normalization
							  )



	NNPolicy.train(state, action)

	## save model

	# NNPolicy.model.save('NNpolicy_weights.h5')
	print('end')



def main():

	import argparse
	parser = argparse.ArgumentParser ()
	parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
	# Experiment meta-params
	parser.add_argument('--exp_name', type=str, default='mb_mpc')
	parser.add_argument('--seed', type=int, default=3)
	parser.add_argument('--render', action='store_true')
	# Training args
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
	parser.add_argument('--onpol_iters', '-n', type=int, default=10)	#Aggregation iters  5
	parser.add_argument('--dyn_iters', '-nd', type=int, default=20)  	#epochs
	parser.add_argument('--batch_size', '-b', type=int, default=64)
	# Data collection
	parser.add_argument('--random_paths', '-r', type=int, default=10)  	# random path nums 700
	parser.add_argument('--onpol_paths', '-d', type=int, default=3)		#mpc path nums   30
	parser.add_argument('--ep_len', '-ep', type=int, default=10)		#1000   path length  200 1000
	# Neural network architecture args
	parser.add_argument('--n_layers', '-l', type=int, default=2)
	parser.add_argument('--size', '-s', type=int, default=64)
	# MPC Controller
	parser.add_argument('--mpc_horizon', '-m', type=int, default=10)   		 #mpc simulation H  10
	parser.add_argument('--simulated_paths', '-sp', type=int, default=10)  #mpc  candidate  K 100
	args = parser.parse_args()

	# Set seed
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	# Make data directory if it does not already exist


	# Make env
	if args.env_name is 'HalfCheetah-v2':
		env = HalfCheetahEnvNew()
		cost_fn = cheetah_cost_fn



	env_name = args.env_name #HalfCheetah-v2  My3LineDirect-v1
	cost_fn = cheetah_cost_fn
	env = gym.make(env_name)
	#env.set_goals(45 * 3.14 / 180.0)  # 角度要换成弧度


	logdir = configure_log_dir(logname =env_name, txt='-train')
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
				 n_layers = args.n_layers,
				 size=args.size,
				 activation=tf.nn.relu,
				 output_activation=None,
				 )

if __name__ == "__main__":
	main()