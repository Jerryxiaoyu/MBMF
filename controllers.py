import numpy as np
from cost_functions import trajectory_cost_fn, cheetah_cost_fn
import time
import torch
from torch.autograd import Variable


class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	""" YOUR CODE HERE """

	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """

		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

		act_space = self.env.action_space
		print(act_space)

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		actions = []
		K = self.num_simulated_paths

		observations = [[] for _ in range(self.horizon)]
		next_observations = [[] for _ in range(self.horizon)]
		actions = [[] for _ in range(self.horizon)]
		cost = []
		# sample K sequences of actions

		start = time.time()  # caculate run time --start time point

		path = {}
		for t in range(self.horizon):

			action = []
			for k in range(K):
				action.append(self.env.action_space.sample())
			action = np.array(action)
			# use dynamics model f to generate simulated rollouts
			if t == 0:
				observation = state
				for _ in range(K - 1):
					observation = np.concatenate((observation, state), axis=0)
				next_obs = self.dyn_model.predict(observation, action)

				observations = observation
				actions = action
				next_observations = next_obs
			else:
				observation = next_obs
				next_obs = self.dyn_model.predict(observation, action)

				observations = np.hstack((observations, observation))
				actions = np.hstack((actions, action))
				next_observations = np.hstack((next_observations, next_obs))

		end = time.time()
		runtime1 = end - start
		# print(end-start)
		# evaluate trajectories.

		start = time.time()  # caculate run time --start time point
		a = observations[0].reshape((self.horizon, -1))
		for i in range(K):
			cost.append(trajectory_cost_fn(cheetah_cost_fn, observations[i].reshape((self.horizon, -1)),
										   actions[i].reshape((self.horizon, -1)),
										   next_observations[i].reshape((self.horizon, -1))))


		# find the best one.
		best_index = np.argmin(cost)
		# return best a0
		best_action = actions[best_index][0:self.env.action_space.shape[0]]

		end = time.time()
		runtime2 = end - start
		# print(end-start)
		return best_action

	def get_action2(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		actions = []
		K = self.num_simulated_paths

		start = time.time()  # caculate run time --start time point
		observations = [[] for _ in range(K)]
		next_observations = [[] for _ in range(K)]
		cost = []
		# sample K sequences of actions
		for k in range(K):
			action = []
			for _ in range(self.horizon):
				action.append(self.env.action_space.sample())
			action = np.array(action)
			actions.append(action)

			# use dynamics model f to generate simulated rollouts
			for t in range(self.horizon):
				if t == 0:
					observations[k] = state
					next_obs = self.dyn_model.predict(observations[k], actions[k][t])
					next_observations[k] = next_obs
				else:
					observations[k] = np.row_stack((observations[k], next_observations[k][t - 1]))
					next_obs = self.dyn_model.predict(observations[k][t], actions[k][t])
					next_observations[k] = np.row_stack((next_observations[k], next_obs))
			# evaluate trajectories.
			cost.append(trajectory_cost_fn(cheetah_cost_fn, observations[k], actions[k], next_observations[k]))

		end = time.time()
		runtime1 = end - start

		# find the best one.
		best_index = np.argmin(cost)
		# return best a0
		best_action = actions[best_index][0]

		return best_action

	def get_action3(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """

		K = self.num_simulated_paths

		start = time.time()  # caculate run time --start time point

		s0 = torch.Tensor(1, 20).uniform_(-1, 1).double()
		s0 = torch.from_numpy(state).double()

		obs = torch.zeros(self.horizon, K, 20).double()
		next_obs = torch.zeros(self.horizon, K, 20).double()
		act = torch.Tensor(self.horizon, K, 6).uniform_(-1, 1).double()
		obs_v = Variable(obs).cuda()
		next_obs_v = Variable(obs).cuda()
		act_v = Variable(act).cuda()
		for i in range(K):
			obs_v[0, i] = s0

		for i in range(self.horizon-1):
			next_obs_v[i]	 = self.dyn_model.predict(obs_v[i],act_v[i])+ obs_v[i]
			obs_v[i + 1] = next_obs_v[i]
		end = time.time()
		runtime1 = end - start

	def get_action4(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """

		K = self.num_simulated_paths
		act_dim = self.env.action_space.shape[0]
		cost =[]
		start = time.time()  # caculate run time --start time point

		state = torch.from_numpy(state).double().cuda()

		# The following script is too low because of using env.action_space

		# actions= []
		# for k in range(K *self.horizon):
		# 	actions = np.append(actions,self.env.action_space.sample())
		# actions = actions.reshape((K, -1))
		# actions = torch.from_numpy(actions).double().cuda()

		## 存在隐患! 当action sapce != (-1 1)
		actions =torch.Tensor(K,self.horizon* act_dim).uniform_(-1, 1).double().cuda()

		end = time.time()
		runtime1 = end - start
		for t in range(self.horizon):

			action = actions[:, t*act_dim:(t+1)*act_dim]
			# use dynamics model f to generate simulated rollouts
			if t == 0:
				observation = state
				for _ in range(K - 1):
					observation = torch.cat((observation, state), 0)
				next_obs = self.dyn_model.predict(observation, action)

				observations = observation
				#actions = action
				next_observations = next_obs
			else:
				observation = next_obs
				next_obs = self.dyn_model.predict(observation, action)

				observations = torch.cat((observations, observation),1)
				#actions = torch.cat((actions, action),1)
				next_observations = torch.cat((next_observations, next_obs),1)

		end2 = time.time()
		runtime2 = end2 - start
		start = time.time()  # caculate run time --start time point

		observations = observations.cpu().numpy()
		actions = actions.cpu().numpy()
		next_observations = next_observations.cpu().numpy()


		for i in range(K):
			cost.append(trajectory_cost_fn(cheetah_cost_fn, observations[i].reshape((self.horizon, -1)),
										   actions[i].reshape((self.horizon, -1)),
										   next_observations[i].reshape((self.horizon, -1))))


		# find the best one.
		best_index = np.argmin(cost)
		# return best a0
		best_action = actions[best_index,0:act_dim]

		end = time.time()
		runtime3= end - start
		# print(end-start)
		return best_action







