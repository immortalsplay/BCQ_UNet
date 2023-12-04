import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from UNet import FeatureExtractor, cropping

def add_bonus_for_close_agent(reward, state, target_position, long_threshold_distance=0.8, short_threshold_distance=0.2, bonus_reward=0.5):
	"""
	Adds a bonus reward to agents that are within a certain threslong_hold distance from the target.

	:param reward: (torch.Tensor) The original rewards for each agent in the batch.
	:param state: (torch.Tensor) The states for each agent in the batch.
	:param target_position: (numpy.ndarray or torch.Tensor) The target's position.
	:param threshold_distance: (float) The distance within which to give the bonus reward.
	:param bonus_reward: (float) The amount of bonus reward to give.
	:return: (torch.Tensor) The updated rewards with the bonus included.
	"""
	# Ensure target_position is a PyTorch tensor and on the same device as the reward
	if not isinstance(target_position, torch.Tensor):
		target_position = torch.tensor(target_position, device=reward.device, dtype=torch.float32)

	# Calculate the distances
	agent_positions = state[:, :2]  # Assuming the first two dimensions of state are the agent's position
	distances = torch.norm(agent_positions - target_position, dim=1, p=2)

	# Find agents that are within the threshold distance and add the bonus reward
	long_bonus_indices = distances <= long_threshold_distance
	short_bonus_indices = distances <= short_threshold_distance
	# 增加对长距离的奖励
	reward[long_bonus_indices] += bonus_reward
	# 然后额外增加对短距离的奖励
	reward[short_bonus_indices] += bonus_reward

	# reward = reward.type(torch.int)
	
	return reward

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

#MNetwork same as Critic
class MNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        m1 = F.relu(self.l1(torch.cat([state, action], 1)))
        m1 = F.relu(self.l2(m1))
        m1 = self.l3(m1)

        m2 = F.relu(self.l4(torch.cat([state, action], 1)))
        m2 = F.relu(self.l5(m2))
        m2 = self.l6(m2)
        return m1, m2

    def m1(self, state, action):
        m1 = F.relu(self.l1(torch.cat([state, action], 1)))
        m1 = F.relu(self.l2(m1))
        m1 = self.l3(m1)
        return m1

# initial VNetwork without double Q 
class VNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(VNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        v = F.relu(self.l1(torch.cat([state, action], 1)))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, env, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
	# def __init__(self, state_dim, action_dim, max_action, device,  discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		# UNet & MLP parameters	
		unet_params = {'n_channels': 3, 'n_classes': 64}
		mlp_params = {'input_dim': 640000, 'output_dim': 64}  # 根据您的实际情况调整

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
		self.critic_losses = []

		#MNetwork
		self.mnetwork = MNetwork(state_dim, action_dim).to(device)
		self.m_target_network = copy.deepcopy(self.mnetwork)
		self.mnetwork_optimizer = torch.optim.Adam(self.mnetwork.parameters(), lr=1e-3)
		self.m_losses = []
		# #VNetwork
		# self.vnetwork = VNetwork(state_dim, action_dim).to(device)
		# self.v_target_network = copy.deepcopy(self.vnetwork)
		# self.vnetwork_optimizer = torch.optim.Adam(self.vnetwork.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.feature_extractor = FeatureExtractor(unet_params, mlp_params).to(device)
		self.feature_extractor_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-3)

		self.env = env

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, target_position,iterations,  batch_size=100, m_train_frequency=5):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
			frames = []
			for i in range(batch_size):		
				single_state = state[i]
				# 然后将单个状态转换为CPU上的NumPy数组
				single_state = single_state.cpu()
				# print("state.shape", single_state.shape)
				cropped_image = cropping(self.env, single_state)
				cropped_image = cropped_image.copy()  # 确保没有负步长
				frames.append(cropped_image)
			
			cropped_images_np = np.stack(frames)
			cropped_images_tensor = torch.from_numpy(cropped_images_np).float()
			cropped_images_tensor = cropped_images_tensor.permute(0, 3, 1, 2)  
			cropped_images_tensor = cropped_images_tensor.to(self.device)

			# FeatureExtractor Training
			self.feature_extractor.train()  # train mode for dropout
			mlp_output = self.feature_extractor(cropped_images_tensor)

			#bonus reward
			# reward = add_bonus_for_close_agent(reward, state, target_position)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				max_q_targets = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				max_q_targets = max_q_targets.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				# target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# # Take max over each action sampled from the VAE
				# target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * max_q_targets

				# Compute value of perturbed actions sampled from the VAE

				# Compute M-Network Need vae? 
				
				target_M1, target_M2 = self.m_target_network(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double M-learning 
				max_M_targets = self.lmbda * torch.min(target_M1, target_M2) + (1. - self.lmbda) * torch.max(target_M1, target_M2)
				# Take max over each action sampled from the VAE
				max_M_targets = max_M_targets.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

			target_M = reward**2 + not_done*2*self.discount*reward*max_q_targets\
								+ not_done*(self.discount**2)*max_M_targets
				
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			self.critic_losses.append(critic_loss.item())
			# M update here
			if it % m_train_frequency == 0:
				current_M1, current_M2 = self.mnetwork(state, action)
				# actual_var = compute_var(state, action)
				m_loss = F.mse_loss(current_M1, target_M) + F.mse_loss(current_M2, target_M)

				self.mnetwork_optimizer.zero_grad()
				m_loss.backward()
				self.mnetwork_optimizer.step()
				self.m_losses.append(m_loss.item())
			# #V update here
			current_Q1, current_Q2 = self.critic(state, action)
			current_M1, current_M2 = self.mnetwork(state, action)
			# # max_cur_M = torch.min(current_M1, current_M2)
			# # max_cur_Q = torch.min(current_Q1, current_Q2)

			# v_targets1 = current_M1 - current_Q1.pow(2)
			# v_targets2 = current_M2 - current_Q2.pow(2)
			
			# current_V = self.vnetwork(state, action)
			# v_loss = F.mse_loss(current_V, v_targets1) + F.mse_loss(current_V, v_targets2)

			# # # v_targets = (v_targets1 + v_targets2) / 2
			# # # v_loss = F.mse_loss(current_V, v_targets)

			# self.vnetwork_optimizer.zero_grad()
			# v_loss.backward()
			# self.vnetwork_optimizer.step()

			# v_loss1 = F.mse_loss(current_V, v_targets1)
			# self.vnetwork_optimizer.zero_grad()
			# v_loss1.backward()  # Retain the computation graph

			# # v_loss2 = F.mse_loss(current_V, v_targets2)
			# # # No need to zero_grad() if you are accumulating gradients
			# # v_loss2.backward()  # Now you can backward through the graph a second time

			# self.vnetwork_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.mnetwork.parameters(), self.m_target_network.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# for param, target_param in zip(self.vnetwork.parameters(), self.v_target_network.parameters()):
			# 	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)