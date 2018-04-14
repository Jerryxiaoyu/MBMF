import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import torch.nn.functional as F
from sklearn import preprocessing
import logger
torch.set_default_tensor_type('torch.DoubleTensor')


tensor_logger = logger.Logger('./logs')

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


class Normalization():
	"""
	Write a function to take in a dataset and compute the means, and stds.
	Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions

	X_scaled = scaler.transform(X)
	X_inv=scaler.inverse_transform(X_scaled)
	"""
	""" YOUR CODE HERE """
	def __init__(self,data):
		self.scaler = [[],[]]
		self.scaler[0]= np.mean(data, axis=0)
		self.scaler[1]= np.sqrt(np.var(data,axis=0))

		self.scaler[1] = np.where(self.scaler[1] == 0, 1, self.scaler[1])

	def transform (self, data):
		x = (data - self.scaler[0])/self.scaler[1]
		y = np.nan_to_num(x)
		return y
	def inverse_transform(self,data):
		return data*self.scaler[1]+self.scaler[0]




class Dataset(Dataset):
	"""
	Motion dataset.
	"""

	def __init__(self, x, y):
		self.x_data = torch.from_numpy(x).double()
		self.y_data = torch.from_numpy(y).double()
		self.len = x.shape[0]

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


class NNDynamicsModel(torch.nn.Module):

	def __init__(self,
				 env,
				 hidden_size=(500, 500),
				 activation='relu',

				 ):
		"""
		In the constructor we instantiate two nn.Linear module
		"""

		super(NNDynamicsModel, self).__init__()

		obs_dim = env.observation_space.shape[0]
		act_dim = env.action_space.shape[0]
		self.input_dim = obs_dim + act_dim
		self.ouput_dim = obs_dim


		if activation == 'relu':
			self.activation = F.relu
		elif activation == 'tanh':
			self.activation = F.tanh

		self.dropout = torch.nn.Dropout(p=0.2)

		self.affine_layers = torch.nn.ModuleList()
		last_dim = self.input_dim
		for nh in hidden_size:
			self.affine_layers.append(torch.nn.Linear(last_dim, nh))
			last_dim = nh

		self.action_mean = torch.nn.Linear(last_dim, self.ouput_dim)




	def forward(self, x):
		"""
		In the forward function we accept a Variable of input data and we must return
		a Variable of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Variables.
		"""
		for affine in self.affine_layers:
			x = self.activation(affine(x))
			x= self.dropout(x)

		y_pred = self.action_mean(x)
		return y_pred

	def fit(self, x, y, epoch_size=20, batch_size=512,test = False):

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		if test is True:
			train_len = int(x.shape[0] *0.8)
			test_loader = Dataset(x[train_len:], y[train_len:])
		else:
			train_len =x.shape[0]

		scaler_x = Normalization(x[:train_len])
		scaler_y = Normalization(y[:train_len])

		data_x = scaler_x.transform(x)
		data_y = scaler_y.transform(y)

		self.scaler_cuda_x_mean = torch.from_numpy(scaler_x.scaler[0]).cuda()
		self.scaler_cuda_x_scaler = torch.from_numpy(scaler_x.scaler[1]).cuda()
		self.scaler_cuda_y_mean = torch.from_numpy(scaler_y.scaler[0]).cuda()
		self.scaler_cuda_y_scaler = torch.from_numpy(scaler_y.scaler[1]).cuda()

		dataset = Dataset(data_x, data_y)
		train_loader = DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=2)



		dataset_size = train_loader.sampler.data_source.len
		# Training loop
		for epoch in range(epoch_size):
			for i, data in enumerate(train_loader, 0):
				# get the inputs
				inputs, labels = data

				# wrap them in Variable
				inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

				# Forward pass: Compute predicted y by passing x to the model
				y_pred = self(inputs)

				# Compute and print loss
				loss = criterion(y_pred, labels)

				# show for debug
				if i == 0:
					print('Epoch ', (epoch + 1), '/', epoch_size, '-----------------------------')
				if int(dataset_size / batch_size) >= 10:
					if i in np.linspace(0, int(dataset_size / batch_size), num=10, dtype=int):
						print('  ', batch_size * (i + 1), '/', dataset_size, ' - loss: %.3f' % loss.data[0])

					# print('  ', batch_size * (i + 1), '/', dataset_size, '- loss: %.3f' % loss.data[0])


				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# ============ TensorBoard logging ============#
				# (1) Log the scalar values
				info = {
					'loss': loss.data[0],
					# 'accuracy': accuracy.data[0]
				}

				for tag, value in info.items():
					tensor_logger.scalar_summary(tag, value, i + 1)

				# (2) Log values and gradients of the parameters (histogram)
				for tag, value in self.named_parameters():
					tag = tag.replace('.', '/')
					tensor_logger.histo_summary(tag, value.data.cpu().numpy(), i + 1)

			if test is True:
				for i, data in enumerate(test_loader, 0):
					# get the inputs
					inputs, labels = data
					# wrap them in Variable
					inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
					# Forward pass: Compute predicted y by passing x to the model
					y_pred = self(inputs)

					# Compute and print loss
					loss = criterion(y_pred, labels)

					# ============ TensorBoard logging ============#
					# (1) Log the scalar values
					info = {
						'Val-loss': loss.data[0],
						# 'accuracy': accuracy.data[0]
					}

					for tag, value in info.items():
						tensor_logger.scalar_summary(tag, value, i + 1)


				print('Epoch ', (epoch + 1), '/', epoch_size,
					  'Validation loss %.3f' % loss.data[0])


	def predict(self, state, action):
		"""
		Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model

		:param states:    numpy array (lengthOfRollout , state.shape[1] )
		:param actions:
		:return:
		"""
		""" YOUR CODE HERE """


		x_data =torch.cat((state, action), 1)
		x_data = (x_data - self.scaler_cuda_x_mean) / self.scaler_cuda_x_scaler

		x_data = Variable(x_data)

		f = self.forward(x_data)
		f = f.data * self.scaler_cuda_y_scaler+self.scaler_cuda_y_mean
		next_states = f + state



		return next_states
