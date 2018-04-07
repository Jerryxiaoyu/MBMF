import gym
import numpy as np
from datetime import datetime
from utils import Logger, Scaler
import csv
import matplotlib.pyplot as plt
import torch
from utils import Logger, configure_log_dir
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,SGD,Adam,Adagrad
from keras.models import load_model
from keras.losses import mean_squared_error

from sklearn import preprocessing

def compute_normalization(data):
	"""
	Write a function to take in a dataset and compute the means, and stds.
	Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions

	X_scaled = scaler.transform(X)
	X_inv=scaler.inverse_transform(X_scaled)
	"""

	""" YOUR CODE HERE """
	scaler = preprocessing.StandardScaler().fit(data)
	return scaler

def run_RandomOnce(env):
	""" Run single episode with random state

		Args:
			env: ai gym environment

		Returns: 4-tuple of NumPy arrays
			observes_1: shape = (episode len, obs_dim)
			actions: shape = (episode len, act_dim)
			rewards: shape = (episode len,)
			observes_2: shape = (episode len, obs_dim)
		"""
	obs = env.reset()
	observes_1, actions, rewards, observes_2 = [], [], [],[]

	obs = obs.astype(np.float64).reshape((1, -1))
	observes_1.append(obs)
	action = env.action_space.sample()
	action = action.astype(np.float64).reshape((1, -1))
	actions.append(action)
	obs, reward, done, _ = env.step(action)
	obs = obs.astype(np.float64).reshape((1, -1))
	observes_2.append(obs)
	if not isinstance(reward, float):
		reward = np.asscalar(reward)
	rewards.append(reward)

	return (np.concatenate(observes_1), np.concatenate(actions),
			np.array(rewards, dtype=np.float64),np.concatenate(observes_2))

def run_rollout(env, LengthOfRollout):
	""" Run  rollout with random state

			Args:
				env: ai gym environment
				LengthOfRollout: length of the rollout

			Returns: 4-tuple of NumPy arrays
				observes_1: shape = (episode len, obs_dim)
				actions: shape = (episode len, act_dim)
				rewards: shape = (episode len,)
				observes_2: shape = (episode len, obs_dim)
	"""

	obs = env.reset()
	observes_1, actions, rewards, observes_2 = [], [], [],[]
	for _ in range(LengthOfRollout):
		obs = obs.astype(np.float64).reshape((1, -1))
		observes_1.append(obs)
		action = env.action_space.sample()
		action = action.astype(np.float64).reshape((1, -1))
		actions.append(action)
		obs, reward, done, _ = env.step(action)
		obs = obs.astype(np.float64).reshape((1, -1))
		observes_2.append(obs)
		if not isinstance(reward, float):
			reward = np.asscalar(reward)
		rewards.append(reward)

	return (np.concatenate(observes_1), np.concatenate(actions),
			np.array(rewards, dtype=np.float64),np.concatenate(observes_2))


def run_rolloutScaler(env,scaler, LengthOfRollout):
	""" Run  rollout with random state

			Args:
				env: ai gym environment
				LengthOfRollout: length of the rollout

			Returns: 4-tuple of NumPy arrays
				observes_1: shape = (episode len, obs_dim)
				actions: shape = (episode len, act_dim)
				rewards: shape = (episode len,)
				observes_2: shape = (episode len, obs_dim)
	"""
	scaler_action =Scaler(env.action_space.shape[0])
	scaler_act, scaler_offset= scaler_action.get()
	scale, offset = scaler.get()
	scale[-1] = 1.0  # don't scale time step feature
	offset[-1] = 0.0  # don't offset time step feature
	obs = env.reset()
	observes_1, actions, rewards, observes_2 = [], [], [], []
	for _ in range(LengthOfRollout):
		obs = obs.astype(np.float64).reshape((1, -1))
		observes_1.append(obs)
		action = env.action_space.sample()
		action = action.astype(np.float64).reshape((1, -1))

		actions.append(action)
		obs, reward, done, _ = env.step(action)
		obs = obs.astype(np.float64).reshape((1, -1))
		observes_2.append(obs)

		if not isinstance(reward, float):
			reward = np.asscalar(reward)
		rewards.append(reward)

	return (np.concatenate(observes_1), np.concatenate(actions),
			np.array(rewards, dtype=np.float64), np.concatenate(observes_2))

def collect_data(env, datasize, Rollout_Collect = 'rollout', LengthOfRollout =100):
	""" collect data
			   Args:
				   env: ai gym environment
				   datasize: the number of total data
				   Rollout_Collect : if choose rollout_collect, then True, else chose randonce_collect
				   LengthOfRollout: length of the rollout

			   Returns:
				   True
	   """
	trajectory = {}
	now = datetime.now().strftime("%b-%d_%H:%M:%S")  # create unique directories  格林尼治时间!!!  utcnow改为now
	logger = Logger(logname=env_name, now=now)
	print('Collection is processing:')
	if Rollout_Collect != 'randonce':
		if datasize % LengthOfRollout ==0:
			Num_itr = int(datasize/LengthOfRollout)
		else:
			return print('Datasize is not devided by LengthOfRollout!')
	else:
		Num_itr = datasize
	for num in range(Num_itr):
		if Rollout_Collect=='rollout':
			observes_1, actions, rewards, observes_2 = run_rollout(env,LengthOfRollout)
		elif Rollout_Collect=='randonce':
			observes_1, actions, rewards, observes_2 = run_RandomOnce(env)
		elif Rollout_Collect=='rolloutScaler':
			obs_dim = env.observation_space.shape[0]  # 20
			scaler = Scaler(obs_dim)
			observes_1, actions, rewards, observes_2 = run_rolloutScaler(env,scaler,LengthOfRollout)
			observes_1s = scaler.preprocess(observes_1)
			observes_2s = scaler.preprocess(observes_2)
		else:
			print("Param 'Rollout_Collect' wasn't given !")

		x_data = np.concatenate((observes_1, actions), axis=1)
		y_data = observes_2 - observes_1
		obs_orig = np.concatenate((observes_1, observes_2), axis=1)
		data = np.concatenate((x_data, y_data), axis=1)
		for j in range(data.shape[0]):
			for i in range(data.shape[1]):
				trajectory[i] = data[j][i]
			logger.log(trajectory)
			logger.write(display=False)
		#Completion info
		if  (num+1) % ( Num_itr / 10) == 0:
			print('Data has been being collected... %d%% '%((num+1)*100/Num_itr))
	logger.close()
	print('Collection is completed!\n')
	return True


def load_data(data_name, data_num, test_percentage =0.2):
	'''
	取数据集中的前80%作为训练集,后20%为测试集合

	:param data_name: name of the data file
	:param data_num:  number of the data
	:return: (x_train,y_train),(x_test,y_test)

	load_data('logR1000L50.csv', data_num =50000)
	'''

	f = open('data/'+data_name, 'r')
	row = csv.reader(f, delimiter=",")
	n_row = 0
	Num_data = data_num             #number of total data
	Num_val =int( data_num * test_percentage )      #Validation set
	x_train,y_train,x_test,y_test =[],[],[],[]
	for i in range(Num_data):
		x_train.append([])
		y_train.append([])

	for r in row:
		if n_row != 0:
			for i in range(26):
				x_train[(n_row - 1) % Num_data].append(float(r[i]))
			for i in range(26, 46):
				y_train[(n_row - 1) % Num_data].append(float(r[i]))
		n_row = n_row + 1
	f.close()

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_test = x_train[Num_data-Num_val:]
	y_test = y_train[Num_data-Num_val:]
	x_train = np.delete(x_train, range(Num_data - Num_val, Num_data), 0)
	y_train = np.delete(y_train, range(Num_data - Num_val, Num_data), 0)
	print('Data is loaded!\n')
	print('x_train shape', x_train.shape)
	print('y_train shape', y_train.shape)
	print('x_test shape', x_test.shape)
	print('y_test shape', y_test.shape)

	return (x_train,y_train),(x_test,y_test)



def predict_error_scaled(model, x_test,y_test,lengthOfRollout = 1000):
	'''
	对测试集进行误差预测
	:param x_test:
	:param y_test:
	:return:
	'''
	logger_flag = False

	#lengthOfRollout =x_test.shape[0]  #预测误差数据的长度

	[states_test,actions_test]= np.split(x_test, [20],axis=1)

	states = torch.from_numpy(states_test).double().cuda()
	actions = torch.from_numpy(actions_test).double().cuda()
	observation = torch.zeros((lengthOfRollout,states.shape[1])).cuda()
	for t in range(lengthOfRollout):

		# use dynamics model f to generate simulated rollouts
		if t == 0:
			observation[t] = states[0:1]

		else:
			observation[t] = next_obs

		next_obs = model.predict(observation[t:t+1], actions[t:t+1])

	states_eval = observation.cpu().numpy()
	if logger_flag is True:
		trajectory = {}
		now = datetime.now().strftime("%b-%d_%H:%M:%S")  # create unique directories  格林尼治时间!!!  utcnow改为now
		logger = Logger('predict', now=now)
		for j in range(states_eval.shape[0]):
			for i in range(states_eval.shape[1]):
				trajectory[i] = states_eval[j][i]
			logger.log(trajectory)
			logger.write(display=False)
		logger.close()
	print('Prediction is finished !')

	mse = np.mean((states_test[0:lengthOfRollout] - states_eval) ** 2)

	print('eorror = ',mse)
	print('y_predict =', states_eval[1])
	print('y_true =',  states_test[1] )

	return states_eval



env_name = 'HalfCheetah-v2'
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]  #20
act_dim = env.action_space.shape[0]  #6



DataNum =50000



#collect_data(env, DataNum, Rollout_Collect='rollout',LengthOfRollout =1000)


# # train set
# (x_train,y_train),(x_test,y_test) = load_data('log-R50L1000.csv',test_percentage = 0.2, data_num =50000)
# #scaler
# scaler_x = compute_normalization(x_train)
# scaler_y = compute_normalization(y_train)
# normalization = [scaler_x, scaler_y]

#train(x_train,y_train,x_test,y_test,model_name = 'my_model_scaled_01181405_001')

# test set
#(x_train,y_train),(x_test,y_test) = load_data('log-test1.csv', test_percentage = 1,data_num =1000)
(x_train,y_train),(x_test,y_test) = load_data('log-test1.csv', test_percentage = 1,data_num =1000)
num_predict =1000

# reload model
dyn_model = torch.load('net.pkl')


states_eval = predict_error_scaled(dyn_model,x_test,y_test,lengthOfRollout = num_predict)


# Create log files
logdir = configure_log_dir(logname=env_name, txt='ModelTest')


# save traj of evaluation
logger = Logger(logdir, csvname='log_test' )
trajectory={}
tra_name =['s1-qpos1','s2-qpos2','s3-qpos3','s4-qpos4','s5-qpos5','s6-qpos6','s7-qpos7','s8-qpos8','s9-qvel0','s10-qvel1','s11-qvel2','s12-qvel3','s13-qvel4','s14-qvel5','s15-qvel6','s16-qvel7','s17-qvel8','s18-com0','s19-com1','s20-com2']
for j in range(states_eval.shape[0]):
	for i in range(states_eval.shape[1]):
		trajectory[tra_name[i]] = states_eval[j][i]
	logger.log(trajectory)
	logger.write(display=False)
logger.close()

# save figures of evaluating
state_name = ['s1-qpos1','s2-qpos2','s3-qpos3','s4-qpos4','s5-qpos5','s6-qpos6','s7-qpos7','s8-qpos8','s9-qvel0','s10-qvel1','s11-qvel2','s12-qvel3	','s13-qvel4','s14-qvel5','s15-qvel6','s16-qvel7','s17-qvel8','s18-com0','s19-com1','s20-com2']
LengthOfCurve = 100 # the Length of Horizon in a curve

x = range(LengthOfCurve)
for i in range(20):
	plt.figure()
	plt.plot(x, states_eval[0:LengthOfCurve, i], label="$predict$")
	plt.plot(x, x_test[0:LengthOfCurve:, i], label="$Actual$")

	plt.xlabel("episode")
	plt.ylabel("x(mm)")
	plt.title("Prediction of the state: "+state_name[i])
	plt.legend()
	#plt.show()

	path = os.path.join(logdir,'plot')
	if not (os.path.exists(path)):
		os.makedirs(path)
	plt.savefig(logdir+'/plot/fig'+str(i)+'-'+state_name[i]+'.jpg')
