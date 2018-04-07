import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,SGD,Adam,Adagrad
from keras.models import load_model
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

from utils import Logger
import utils
import os.path as osp, shutil, time, atexit, os, subprocess

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

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
			  output_size,
			  scope,
			  n_layers=2,
			  size=500,
			  activation=tf.tanh,
			  output_activation=None
			  ):
	out = input_placeholder
	with tf.variable_scope(scope):
		for _ in range(n_layers):
			out = tf.layers.dense(out, size, activation=activation)
		out = tf.layers.dense(out, output_size, activation=output_activation)
	return out
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = {'batch':[], 'epoch':[]}
		self.accuracy = {'batch':[], 'epoch':[]}
		self.val_loss = {'batch':[], 'epoch':[]}
		self.val_acc = {'batch':[], 'epoch':[]}

	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.val_acc['batch'].append(logs.get('val_acc'))

	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.val_acc['epoch'].append(logs.get('val_acc'))

	def loss_plot(self, loss_type):

		iters = range(len(self.losses[loss_type]))
		plt.figure()
		# acc
		# plt.plot(iters, self.accuracy[loss_type], 'r', label='train mse')
		# loss
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
		if loss_type == 'epoch':
			# val_acc
			# plt.plot(iters, self.val_acc[loss_type], 'b', label='val mse')
			# val_loss
			plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
		plt.grid(True)
		plt.xlabel(loss_type)
		plt.ylabel('mse-loss')
		plt.legend(loc="upper right")

		# plt.show()
		name = '/main-plot'
		now = datetime.now().strftime("%b-%d_%H-%M-%S")
		plt.title('Loss')

		logdir = utils.LOG_PATH
		if not (os.path.exists(logdir)):
			os.makedirs(logdir)
		path = os.path.join(logdir, 'plot_loss')
		if not (os.path.exists(path)):
			os.makedirs(path)
		#os.makedirs(path)
		plt.savefig(path + name+now + '.jpg')

		logger = Logger(logdir, csvname='log_loss')
		for j in range(len(self.losses[loss_type])):
			logger.log({'itr': iters[j],
						'train-loss': self.losses[loss_type][j],
						'val-loss': self.val_loss[loss_type][j]
						})
			logger.write(display=False)
		logger.close()

class NNDynamicsModel():
	def __init__(self,
				 env,
				 normalization,
				 n_layers=1,
				 size = 500,
				 activation = 'relu',
				 output_activation = 'tanh',
				 batch_size =100,
				 iterations = 30,
				 learning_rate =0.001,
				 ):
		""" YOUR CODE HERE """
		""" Note: Be careful about normalization """
		obs_dim = env.observation_space.shape[0]
		act_dim = env.action_space.shape[0]
		self.input_dim = obs_dim+act_dim
		self.ouput_dim = obs_dim
		self.batch_size= batch_size
		self.iterations = iterations
		self.learning_rate = learning_rate
		self.scaler_x = normalization[0]
		self.scaler_y = normalization[1]

		#model Keras
		self.model = Sequential()
		self.model.add(Dense(input_dim=self.input_dim, units=size, activation=activation))
		for _ in range(n_layers):
			self.model.add(Dense(units=size, activation=activation))
		self.model.add(Dense(units=self.ouput_dim, activation=output_activation))
		self.model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse', 'mae'])



	def fit(self, data_x,data_y):
		"""
		Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
		"""
		"""YOUR CODE HERE """
		self.scaler_x = compute_normalization(data_x)
		self.scaler_y = compute_normalization(data_y)

		data_x = self.scaler_x.transform(data_x)
		data_y = self.scaler_y.transform(data_y)
		history = LossHistory()

		self.model.fit(data_x, data_y,validation_split=0.2, batch_size= self.batch_size, epochs=self.iterations,callbacks=[history]) # call history
		history.loss_plot('epoch')


	def predict(self, states, actions):
		"""
		Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model

		:param states:    numpy array (lengthOfRollout , state.shape[1] )
		:param actions:
		:return:
		"""
		""" YOUR CODE HERE """
		start = time.time()
		a = np.concatenate((states , actions ), axis=1)
		a = self.scaler_x.transform(a)          # scaler
		f = self.model.predict(a,batch_size=1024)
		f = self.scaler_y.inverse_transform(f)  # scaler inv
		next_states = f + states

		end = time.time()
		runtime = end - start
		# print(end-start)
		'''
		H = actions.shape[0]
		
		next_states = np.array([np.zeros(states.shape[1])])
		for t in range(H):
			if t==0:
				a = np.concatenate((states[t],actions[t]),axis=1)
				# a = np.append(states_eval[t - 1], actions[t - 1])
			else:
				a = np.concatenate((next_states[t - 1], actions[t]), axis=1)
			f = self.model.predict(a.reshape((1, -1)))
			next_states = np.row_stack((next_states, f + next_states[t - 1]))

		'''
		return next_states

	def predict2(self, states, actions):
		"""
		Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model

		:param states:    numpy array (lengthOfRollout , state.shape[1] )
		:param actions:
		:return:
		"""
		""" YOUR CODE HERE """

		a = np.concatenate((states.reshape((1, -1)), actions.reshape((1, -1))), axis=1)
		a = self.scaler_x.transform(a)  # scaler
		f = self.model.predict(a,batch_size=32)
		f = self.scaler_y.inverse_transform(f)  # scaler inv
		next_states = f + states

		'''
		H = actions.shape[0]

		next_states = np.array([np.zeros(states.shape[1])])
		for t in range(H):
			if t==0:
				a = np.concatenate((states[t],actions[t]),axis=1)
				# a = np.append(states_eval[t - 1], actions[t - 1])
			else:
				a = np.concatenate((next_states[t - 1], actions[t]), axis=1)
			f = self.model.predict(a.reshape((1, -1)))
			next_states = np.row_stack((next_states, f + next_states[t - 1]))

		'''
		return next_states