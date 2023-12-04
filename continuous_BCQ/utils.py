import numpy as np
import torch
import gym
import d4rl
import argparse
import os

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]


# def collect_d4rl_data(env_name, buffer, num_samples):
#     env = gym.make(env_name)
    
#     dataset = d4rl.qlearning_dataset(env)
    
#     observations = dataset['observations']
#     actions = dataset['actions']
#     rewards = dataset['rewards']
#     next_observations = dataset['next_observations']
#     terminals = dataset['terminals']
    
#     for idx in range(min(len(observations), num_samples)):
#         buffer.add(observations[idx], actions[idx], next_observations[idx], rewards[idx], terminals[idx])
        
#     print(f"Collected data for {min(len(observations), num_samples)} samples from D4RL.")

# parser = argparse.ArgumentParser()
# parser.add_argument("--env", default="maze2d-umaze-v1")
# parser.add_argument("--seed", default=0, type=int)
# parser.add_argument("--buffer_name", default="Default")
# args = parser.parse_args()

# setting = f"{args.env}_{args.seed}"
# buffer_name = f"{args.buffer_name}_{setting}"
# buffer_path = f"./buffers/{buffer_name}"

# # 如果你知道d4rl数据集的大小，可以预先设置缓冲区大小
# buffer = ReplayBuffer(state_dim=4, batch_size=64, device="cuda")

# collect_d4rl_data(args.env, buffer, num_samples=500000)

# buffer.save(buffer_path)

def collect_d4rl_data(env_name, buffer, num_samples):
    env = gym.make(env_name)
    
    dataset = d4rl.qlearning_dataset(env)
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    
    for idx in range(min(len(observations), num_samples)):
        buffer.add(observations[idx], actions[idx], next_observations[idx], rewards[idx], terminals[idx])
        
    print(f"Collected data for {min(len(observations), num_samples)} samples from D4RL.")

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="maze2d-umaze-v1")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--buffer_name", default="Robust")
args = parser.parse_args()

setting = f"{args.env}_{args.seed}"
buffer_name = f"{args.buffer_name}_{setting}"
buffer_path = f"./buffers/{buffer_name}"

# Make sure the buffers directory exists
if not os.path.exists('./buffers/'):
    os.makedirs('./buffers/')

# Get state and action dimensions dynamically from the environment
env = gym.make(args.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, device="cuda")

collect_d4rl_data(args.env, buffer, num_samples=500000)

buffer.save(buffer_path)