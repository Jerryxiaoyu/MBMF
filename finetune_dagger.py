import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import torch.nn.functional as F
from sklearn import preprocessing
import gym
from controllers import MPCcontroller
from cost_functions import cheetah_cost_fn

torch.set_default_tensor_type('torch.DoubleTensor')




env_name = 'HalfCheetah-v2'
Trainset_file = 'data/Train_EXPA02_1.csv'
Testset_file = 'data/Test_EXPA02_1.csv'


STATE_DIM =20
ACTION_DIM =6
hidden_size = (64,64)

#Fisrt train
batch_size=2048
epoch_size =500  #1000
learning_rate =0.0005

#DAGGER
n_episode = 3
steps = 1000        # maximum step for a game
n_epoch =200

#MPC
dyn_model =  torch.load('data/net.pkl')
cost_fn = cheetah_cost_fn
mpc_horizon =15
num_simulated_paths=1000









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




class MotionDataset(Dataset):
	"""
	Motion dataset.
	"""

	def __init__(self,Data_files):
		# 初始化data

		tmp = np.loadtxt(Data_files, dtype=np.str, delimiter=",")
		state =  tmp[1:, 0:STATE_DIM].astype(np.float)
		action = tmp[1:, STATE_DIM:STATE_DIM+ACTION_DIM].astype(np.float)
		# state = np.array(state)
		# action = np.array(action)


		scaler_x = compute_normalization(state)
		scaler_y = compute_normalization(action)
		#
		data_x = scaler_x.transform(state)
		data_y = scaler_y.transform(action)
#		data_y =action

		# data_x = state  #+ np.random.normal(0, 0.001, size =state.shape)
		# data_y = action

		self.x_data = torch.from_numpy(data_x).double()
		self.y_data = torch.from_numpy(data_y).double()
		self.len = state.shape[0]

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

	def Aggregation(self, x, y):

		scaler_x = compute_normalization(x)
		scaler_y = compute_normalization(y)

		x = scaler_x.transform(x)
		y = scaler_y.transform(y)

		self.x_data = torch.cat((self.x_data, torch.from_numpy(x).double()),0)
		self.y_data = torch.cat((self.y_data, torch.from_numpy(y).double()),0)


class Model(torch.nn.Module):

	def __init__(self,state_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
		"""
		In the constructor we instantiate two nn.Linear module
		"""
		log_std =0.01
		super(Model, self).__init__()

		if activation == 'tanh':
			self.activation = F.tanh
		elif activation == 'relu':
			self.activation = F.relu
		elif activation == 'sigmoid':
			self.activation = F.sigmoid

		self.affine_layers =torch. nn.ModuleList()
		last_dim = state_dim
		for nh in hidden_size:
			self.affine_layers.append(torch.nn.Linear(last_dim, nh))
			last_dim = nh

		self.action_mean = torch.nn.Linear(last_dim, action_dim)
		self.action_mean.weight.data.mul_(0.1)
		self.action_mean.bias.data.mul_(0.0)

		self.action_log_std = torch.nn.Parameter(torch.ones(1, action_dim) * log_std)


	def forward(self, x):
		"""
		In the forward function we accept a Variable of input data and we must return
		a Variable of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Variables.
		"""
		for affine in self.affine_layers:
			x = F.tanh(affine(x))

		action_mean = self.action_mean(x)


		return action_mean


	def select_action(self, x):
		#action, _, _ = self.forward(x)
		action_mean= self.forward(x)

		action_log_std = self.action_log_std.expand_as(action_mean)
		action_std = torch.exp(action_log_std)

		action = torch.normal(action_mean, action_std)
		return action

env = gym.make(env_name)

mpc_controller = MPCcontroller(env=env,
							   dyn_model=dyn_model,
							   horizon=mpc_horizon,
							   cost_fn=cost_fn,
							   num_simulated_paths=num_simulated_paths,
							   )


dataset = MotionDataset(Trainset_file)
train_loader = DataLoader(dataset=dataset,
						  batch_size=batch_size,
						  shuffle=True,
						  num_workers=0)

test_dataset = MotionDataset(Testset_file)
test_loader = DataLoader(dataset=dataset,
						  batch_size= test_dataset.len,
						  shuffle=True,
						  num_workers=0)


# our model
model = Model(STATE_DIM, ACTION_DIM, hidden_size=hidden_size, activation='tanh').cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


dataset_size = train_loader.sampler.data_source.len
# Training loop
for epoch in range(epoch_size):
	for i, data in enumerate(train_loader, 0):
		# get the inputs
		inputs, labels = data
		# wrap them in Variable
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model(inputs)

		# Compute and print loss
		loss = criterion(y_pred, labels)
		#show for debug
		if i ==0:
			print('Epoch ',(epoch+1),'/',epoch_size,'-----------------------------')
		if int(dataset_size/batch_size) >= 10:
			if i in np.linspace(0,int(dataset_size/batch_size),num =10, dtype=int):
				print('  ',batch_size*(i+1),'/',dataset_size,'----%.d'%(i),'%%------ - loss: %.3f'%loss.data[0])
		else:
			print('  ', batch_size * (i + 1), '/', dataset_size, '----%.d' % (i), '%%------ - loss: %.3f' % loss.data[0])

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


	for i, data in enumerate(test_loader,0):
		# get the inputs
		inputs, labels = data
		# wrap them in Variable
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
		# Forward pass: Compute predicted y by passing x to the model
		y_pred = model(inputs)

		# Compute and print loss
		loss = criterion(y_pred, labels)
		# show for debug

		print('--------------------------------------Validation results:---------- - loss: %.3f' % loss.data[0])

for param in model.parameters():
	 print(type(param.data), param.size())
	 #print(list(param.data))
print(model)

# # save the net wight
#torch.save(model.state_dict(), 'model/net_params.pkl')  # save only the parameters


batch_size=1024
###===================== Aggregate and retrain

for episode in range(n_episode):
	ob_list = []
	act_list=[]
	# restart the game for every episode
	env = gym.make(env_name)
	ob = env.reset()
	reward_sum = 0.0
	print("#"*50)
	print("# Episode: %d start" % (episode+1))
	for i in range(steps):

		#act = model.predict(img_reshape(ob.img))

		state = Variable(torch.from_numpy(ob.reshape(1,-1)).double()).cuda()

		# Forward pass: Compute predicted y by passing x to the model

		act = model.select_action(state)
		ob, reward, done, _ = env.step(act.data.cpu())
		if done is True:
			break
		else:
			ob_list.append(ob)

		reward_sum += reward
		# print(i, reward, reward_sum, done, str(act[0]))
	print("# step: %d reward: %f " % (i, reward_sum))
	print("#"*50)
	#output_file.write('Number of Steps: %02d\t Reward: %0.04f\n' % (i, reward_sum))


	# if i == (steps-1):
	# 	break

	print("# Episode %d"%(episode+1) , 'is generating best actions....')
	action_all=[]
	for i in range(ob_list.__len__()):
		action = mpc_controller.get_action4(ob_list[i].astype(np.float64).reshape((1, -1)))
		action_all.append(action)
	state_all = np.array(ob_list)
	action_all = np.array(action_all)

	dataset.Aggregation(state_all,action_all)
	train_loader = DataLoader(dataset=dataset,
							  batch_size=batch_size,
							  shuffle=True,
							  num_workers=0)


	dataset_size = train_loader.sampler.data_source.len
	# Training loop
	for epoch in range(epoch_size):
		for i, data in enumerate(train_loader, 0):
			# get the inputs
			inputs, labels = data
			# wrap them in Variable
			inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
			# Forward pass: Compute predicted y by passing x to the model
			y_pred = model(inputs)
			# Compute and print loss
			loss = criterion(y_pred, labels)
			# show for debug
			if i == 0:
				print('Daager Episode: ',(episode+1),'Epoch ', (epoch + 1), '/', epoch_size, '-----------------------------')
			if int(dataset_size / batch_size) >= 10:
				if i in np.linspace(0, int(dataset_size / batch_size), num=10, dtype=int):
					print('  ', batch_size * (i + 1), '/', dataset_size, '----%.d' % (i),
						  '%%------ - loss: %.3f' % loss.data[0])
			else:
				print('  ', batch_size * (i + 1), '/', dataset_size, '----%.d' % (i),
					  '%%------ - loss: %.3f' % loss.data[0])

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	torch.save(model.state_dict(), 'model/net_params'+str(episode)+'.pkl')  # save only the parameters