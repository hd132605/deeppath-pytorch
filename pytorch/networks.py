import torch
from torch import nn
from utils import *


class PolicyNeuralNetTorch(nn.Module):
	def __init__(self):
		super(PolicyNeuralNetTorch, self).__init__()
		fc1 = nn.Linear(state_dim, 512)
		nn.init.xavier_uniform_(fc1.weight)  # TODO L2 regularizer 해야 함. weight_decay 로 넣는다.
		nn.init.constant_(fc1.bias, .0)
		fc2 = nn.Linear(512, 1024)
		nn.init.xavier_uniform_(fc2.weight)
		nn.init.constant_(fc2.bias, .0)
		fc3 = nn.Linear(1024, action_space)
		nn.init.xavier_uniform_(fc3.weight)
		nn.init.constant_(fc3.bias, .0)
		self.linear_relu_stack = nn.Sequential(
			fc1,
			nn.ReLU(),
			fc2,
			nn.ReLU(),
			fc3,
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.linear_relu_stack(x)


class PolicyNeuralNetTorchForBert(nn.Module):
	def __init__(self, applied_state_dim):
		super(PolicyNeuralNetTorchForBert, self).__init__()
		fc1 = nn.Linear(applied_state_dim, 512)
		nn.init.xavier_uniform_(fc1.weight)  # TODO L2 regularizer 해야 함. weight_decay 로 넣는다.
		nn.init.constant_(fc1.bias, .0)
		fc2 = nn.Linear(512, 1024)
		nn.init.xavier_uniform_(fc2.weight)
		nn.init.constant_(fc2.bias, .0)
		fc3 = nn.Linear(1024, action_space)
		nn.init.xavier_uniform_(fc3.weight)
		nn.init.constant_(fc3.bias, .0)
		self.linear_relu_stack = nn.Sequential(
			fc1,
			nn.ReLU(),
			fc2,
			nn.ReLU(),
			fc3,
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.linear_relu_stack(x)

class PolicyNeuralNetAttentionEmbedding(nn.Module):
	def __init__(self):
		super().__init__()
		"""constants"""
		trans_r_dim = 100
		model_dim = 768
		num_heads = 6
		"""initialize <sos>"""
		self.sos = torch.zeros((1, 768))
		torch.nn.init.xavier_uniform_(self.sos)
		"""layers"""
		self.trans_r_layer = nn.Linear(trans_r_dim, model_dim)
		self.attn_encoder = nn.MultiheadAttention(model_dim, num_heads)
		self.attn_decoder = nn.MultiheadAttention(model_dim, num_heads)
		"""initialize layer parameters"""
		nn.init.xavier_uniform_(self.trans_r_layer.weight)
		nn.init.constant_(self.trans_r_layer.bias, .0)

	def forward(self, state):
		"""expected shape
		state.size(): (model_dim * 2,)
		"""
		trans_r_original = state[0]
		cls = state[1:]
		trans_r = self.trans_r_layer(trans_r_original)
		stack = torch.vstack((trans_r, *cls))
		attn1_res, _ = self.attn_encoder(stack, stack, stack)
		attn2_res, _ = self.attn_decoder(self.sos, attn1_res, attn1_res)
		return attn2_res



# 쓰이지 않는 함수
def value_nn(state, state_dim, action_dim):
	activation = nn.ReLU()
	fc1 = nn.Linear(state_dim, 64, bias=False)
	nn.init.kaiming_normal_(fc1.weight)
	fc2 = nn.Linear(64, 1, bias=False)
	nn.init.kaiming_normal_(fc2.weight)

	h1 = activation(fc1(state))
	h2 = activation(fc2(h1))
	value_estimated = torch.squeeze(h2)

	return value_estimated


# 쓰이지 않는 함수
def q_network(state, state_dim, action_dim):
	activation = nn.ReLU()
	fc1 = nn.Linear(state_dim, 128)
	nn.init.kaiming_normal_(fc1.weight)
	fc2 = nn.Linear(128, 64)
	nn.init.kaiming_normal_(fc2.weight)
	fc3 = nn.Linear(64, action_dim)
	nn.init.kaiming_normal_(fc3.weight)

	h1 = activation(fc1(state))
	h2 = activation(fc2(h1))
	h3 = fc3(h2)

	return [fc1.weight, fc2.weight, fc3.weight, h3]

if __name__ == "__main__":
	# policy_nn(torch.rand(1, 200), 200, 3)
	print('networks.py')