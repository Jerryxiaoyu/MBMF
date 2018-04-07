import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import csv
import torch.nn.functional as F
from sklearn import preprocessing

torch.set_default_tensor_type('torch.DoubleTensor')



TotalNum = 100000
Data_files = 'log9.csv'

STATE_DIM =20
ACTION_DIM =6
hidden_size = (500,500)


batch_size=1024
epoch_size =30
learning_rate =0.001

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

	def __init__(self):
		# 初始化data
		state, action = [], []
		for i in range(TotalNum):  ##the number of row!
			state.append([])
			action.append([])

		with open(Data_files, 'r') as f:
			f_csv = csv.reader(f)
			n_row = 0
			for row in f_csv:
				if n_row != 0:  # 第0行沒有資訊
					for i in range(0, STATE_DIM):
						state[n_row - 1].append(float(row[i]))
					for i in range(STATE_DIM, STATE_DIM+ACTION_DIM):
						action[n_row - 1].append(float(row[i]))
				n_row = n_row + 1
		state = np.array(state)
		action = np.array(action)


		scaler_x = compute_normalization(state)
		scaler_y = compute_normalization(action)

		data_x = scaler_x.transform(state)
		data_y = scaler_y.transform(action)

		self.x_data = torch.from_numpy(data_x).double()
		self.y_data = torch.from_numpy(data_y).double()
		self.len = state.shape[0]

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


class Model(torch.nn.Module):

	def __init__(self,state_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
		"""
		In the constructor we instantiate two nn.Linear module
		"""
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

		self.action_log_std = torch.nn.Parameter(torch.zeros(1, action_dim))


	def forward(self, x):
		"""
		In the forward function we accept a Variable of input data and we must return
		a Variable of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Variables.
		"""
		for affine in self.affine_layers:
			x = F.tanh(affine(x))

		y_pred = self.action_mean(x)
		return y_pred


dataset = MotionDataset()
train_loader = DataLoader(dataset=dataset,
						  batch_size=batch_size,
						  shuffle=True,
						  num_workers=0)


# our model
model = Model(STATE_DIM, ACTION_DIM, hidden_size=hidden_size, activation='relu').cuda()

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

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

for param in model.parameters():
	 print(type(param.data), param.size())
	 #print(list(param.data))
print(model)

# # save the net wight
torch.save(model.state_dict(), 'net_params.pkl')  # save only the parameters