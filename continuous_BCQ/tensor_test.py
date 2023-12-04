import numpy as np
import tensorflow as tf
import gym
# import mujoco_py
import d4rl
import torch
import BCQ
import os
import argparse
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from UNet import UNet, MLP, FeatureExtractor

def calculate_distance(agent_position, target_position):
    # 计算两点间的欧几里得距离
    distance = np.linalg.norm(np.array(agent_position) - np.array(target_position))
    return distance

def find_agent_position(img):
    agent_color = np.array([90, 180, 90])
    color_threshold = np.array([10, 10, 10])  # 你可以根据需要调整这个阈值

    lower_bound = agent_color - color_threshold
    upper_bound = agent_color + color_threshold

    mask = np.all((img >= lower_bound) & (img <= upper_bound), axis=-1)
    
    y, x = np.where(mask)
    
    if y.size == 0 or x.size == 0:
        # 如果找不到agent，则返回图像的中心
        return [img.shape[1] / 2, img.shape[0] / 2]
    
    return [np.mean(x), np.mean(y)]

def visualize_policy(policy, env_name, device, num_trials=1, save_path='saved_gifs'):
    env = gym.make(env_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        obs = env.reset()
        done = False
        frames = []

        # env.render()
        img = env.render(mode='rgb_array')

        fig, ax = plt.subplots()

        if hasattr(env, 'viewer') and env.viewer is not None:
            # 设置俯视角度
            env.viewer.cam.elevation = -90  # 俯视角度
            env.viewer.cam.azimuth = 90      # 可选，更改方位角
        while not done:
            # env.render()

            img = env.render(mode='rgb_array')
            frames.append(img)

            ax.imshow(img)

            # 定义agent的位置
            agent_pos = find_agent_position(img) # 这可能需要根据你的环境进行调整

            target_position = env.get_target()
            # state = env._get_obs()  # 获取当前环境状态

            # 假设环境状态包含智能体和目标的x, y坐标
            agent_position = obs[:2]  # 例如: [x_agent, y_agent]

            # 使用先前定义的函数计算距离
            distance = calculate_distance(agent_position, target_position)
            # print("target_position:", target_position)
            print("agent_position:", agent_position)
            # # 打印距离
            # print("Distance between agent and target:", distance)
            # Convert the observation to a tensor and move it to the appropriate device
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action_tensor_or_array = policy.select_action(obs_tensor.detach().cpu().numpy())
            
            # Ensure the action is a numpy array before passing it to env.step
            action = np.array(action_tensor_or_array)

            delta = [action[0]*100, -action[1]*100]

            # 在图像上绘制一个代表动作方向的箭头
            ax.arrow(agent_pos[0], agent_pos[1], delta[0], delta[1], head_width=5, head_length=5, fc='red', ec='red')

            plt.pause(0.01)
            ax.clear()

            obs, reward, done, _ = env.step(action)


        # Save frames as a GIF
        gif_path = os.path.join(save_path, f'trial_{trial + 1}.gif')
        imageio.mimsave(gif_path, frames, duration=0.002) # You can adjust duration as needed

    env.close()
    plt.close()

def compute_q_value(policy, state):
    with torch.no_grad():
        actions = policy.vae.decode(state.unsqueeze(0)).to(policy.device)  
        actions_q = policy.actor_target(state.unsqueeze(0), actions)
        q1, q2 = policy.critic(state.unsqueeze(0), actions_q)

        actions_m = policy.m_target_network(state.unsqueeze(0), actions)
        if isinstance(actions_m, tuple):
            actions_m = torch.cat([actions_m[0], actions_m[1]], dim=1)
        m1, m2 = policy.mnetwork(state.unsqueeze(0), actions_m)
        return torch.min(q1, q2).item() ,torch.min(m1, m2).item()

def transform_coordinates(state, env_bottom_left, env_top_right, plot_bottom_left, plot_top_right):
    # Unpack the environment and plot coordinates
    env_x_min, env_y_min = env_bottom_left
    env_x_max, env_y_max = env_top_right
    plot_x_min, plot_y_max = plot_bottom_left
    plot_x_max, plot_y_min = plot_top_right

    # Calculate the scaling factors
    scale_x = (plot_x_max - plot_x_min) / (env_x_max - env_x_min)
    scale_y = (plot_y_max - plot_y_min) / (env_y_max - env_y_min)

    # Transform the state coordinates
    plot_x = plot_x_min + (state[0] - env_x_min) * scale_x
    plot_y = plot_y_max - (state[1] - env_y_min) * scale_y  # Subtract from plot_y_max because y is inverted in the plot

    return plot_x, plot_y

def cropping(env_name, state ,observation_size = 190):
    # 初始化环境
    
    save_path = "fig"
    img_counter = 0

    env = gym.make(env_name)  

    state = env.reset()

    # obs = env.reset()
    done = False
    frames = []

    # env.render()
    # img = env.render(mode='rgb_array')

    fig, ax = plt.subplots()

    if hasattr(env, 'viewer') and env.viewer is not None:
        # 设置俯视角度
        env.viewer.cam.elevation = -90  # 俯视角度
        env.viewer.cam.azimuth = 90      # 可选，更改方位角

    # while not done:
    qpos = np.array([state[0], state[1]])
    qvel = np.array([state[2], state[3]])
    env.set_state(qpos, qvel)


    img = env.render(mode='rgb_array')
    # 计算agent的图像坐标
    plot_x, plot_y = transform_coordinates(
        qpos,
        env_bottom_left=(-0.107, 0.13),
        env_top_right=(3.72, 3.72),
        plot_bottom_left=(156, 345),
        plot_top_right=(345, 156)
    )
    # print('state[:2]',state[:2])
    # print('plot_x, plot_y',plot_x, plot_y)

    # 将坐标转换为整数
    plot_x = int(plot_x)
    plot_y = int(plot_y)

    # 计算裁剪区域
    half_size = observation_size // 2
    top = max(0, plot_y - half_size - 22)
    bottom = min(img.shape[0], plot_y + half_size - 22)
    left = max(0, plot_x - half_size+6)
    right = min(img.shape[1], plot_x + half_size+6)


    cropped_img = img[top:bottom, left:right]

    return cropped_img
    # 绘制裁剪后的图像
    # plt.imshow(cropped_img)
    # plt.pause(0.01)  # 暂停一会儿，以便可以看到图像

    # file_name = f"{save_path}cropped_img_{img_counter}.png"
    # plt.savefig(file_name)
    # print(f"Image saved: {file_name}")

if __name__ == "__main__":
    ############################# dataset #############################
    # actions = np.load('buffers/Robust_maze2d-umaze-v1_0_action.npy')
    states = np.load('buffers/Robust_maze2d-umaze-v1_0_state.npy')
    # rewards = np.load('buffers/Robust_maze2d-umaze-v1_0_reward.npy')

    # minstate = np.min(states[:, 1],axis=0)
    # print('minstate',minstate)
    # # Find the maximum y value in the states
    # y_min = 0.8
    # y_max = 1.3
    # x_min = 0.8
    # x_max = 1.3
    # # 筛选出在此 y 值范围内的状态
    # filtered_states = states[(states[:, 0] > x_min) & (states[:, 0] < x_max) & (states[:, 1] >= y_min) & (states[:, 1] <= y_max)]

    # # 计算在此范围内的状态数量
    # num_states_in_range = len(filtered_states)
    # print("Number of states with y in the range 3 to 3.2:", num_states_in_range)
    # # # Assume the states are represented by their (x, y) coordinates
    # # # Define the range for the upper left corner
    # # # For example, if the upper left corner is defined by x < 0.5 and y > 0.5
    # # x_upper_left_max = 0.7
    # # y_upper_left_min = 2.9

    # # # Filter for states in the upper left corner
    # # # This assumes that the first two dimensions in the states array are x and y coordinates
    # upper_left_states = states[(states[:, 0] > x_min) & (states[:, 0] < x_max) & (states[:, 1] >= y_min) & (states[:, 1] <= y_max)]
    # upper_left_actions = actions[(states[:, 0] > x_min) & (states[:, 0] < x_max) & (states[:, 1] >= y_min) & (states[:, 1] <= y_max)]

    # env = gym.make('maze2d-umaze-v1')

    # # 定义地图上的点，你需要根据实际情况来确定这些点
    # obs = env.reset()
    # done = False
    # frames = []

    # # env.render()
    # img = env.render(mode='rgb_array')

    # fig, ax = plt.subplots()

    # if hasattr(env, 'viewer') and env.viewer is not None:
    #     # 设置俯视角度
    #     env.viewer.cam.elevation = -90  # 俯视角度
    #     env.viewer.cam.azimuth = 90      # 可选，更改方位角

    # img = env.render(mode='rgb_array')
    # frames.append(img)

    # ax.imshow(img)    
    # # Define four arbitrary points within the upper left corner range
    # # Note: Adjust the action range according to your environment's action space
    # maxstate = np.max(states[:, 1],axis=0)
    # print('maxstate',maxstate)

    # indices = ((states[:, 0] > x_min) & (states[:, 0] < x_max) & (states[:, 1] >= y_min) & (states[:, 1] <= y_max))
    # filtered_states = states[indices]
    # filtered_actions = actions[indices]
    # filtered_rewards = rewards[indices]

    # indexes = np.where(filtered_actions[:, 1] > 0)
    # y_up_reward = filtered_rewards[indexes]

    # # Assuming bg_img is already loaded from the environment
    # # Create a matplotlib figure and axis
    # if len(filtered_states) >= 4:
    #     selected_indices = np.random.choice(len(filtered_states), 4, replace=False)
    #     # Proceed with the rest of your code for plotting
    # else:
    #     print("Not enough states in the dataset after filtering to select 4 unique points.")
    # # Draw arrows for the filtered actions at the filtered states
    # for idx in selected_indices:
    #     state = filtered_states[idx]
    #     action = filtered_actions[idx]
    #     delta = [action[0]*30, -action[1]*30]
    #     plot_x, plot_y = transform_coordinates(
    #     state[:2],
    #     env_bottom_left=(0, 0),
    #     env_top_right=(3.72, 3.72),
    #     plot_bottom_left=(156, 345),
    #     plot_top_right=(345, 156)
    # )
    #     print('state',state)
    #     print('plot_x, plot_y',plot_x, plot_y)
    #     print('action',action)
    #     # Adjust the arrow scale if necessary
    #     ax.arrow(plot_x, plot_y, delta[0], delta[1], head_width=5, head_length=5, fc='red', ec='red')

    # # Show the plot
    # plt.show()

    # # Close the environment
    # env.close()

    ############################# matrix visualization #############################
    # # Initialize your environment, policy and device
    # env = gym.make('maze2d-umaze-v1')
    # env.reset()
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0] 
    # max_action = float(env.action_space.high[0])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # policy = BCQ.BCQ(state_dim, action_dim, max_action, device)  # add other arguments as needed

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env", default="maze2d-umaze-v1")
    # # parser.add_argument("--env", default="Hopper-v3")               # OpenAI gym environment name
    # parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    # parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate

    # parser.add_argument("--max_timesteps", default=1e5, type=int)
    
    # # parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    # parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
    # parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    # parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    # parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    # parser.add_argument("--discount", default=0.99)                 # Discount factor
    # parser.add_argument("--tau", default=0.005)                     # Target network update rate
    # parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    # parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    # parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    # parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
    # args = parser.parse_args()

    # setting = f"{args.env}_{args.seed}"
    # # Load saved weights
    
    # iteration = "970000.0"  # Replace with your specific iteration

    # # Construct paths based on the specified setting and iteration
    # actor_path = f"./models_M2/BCQ_actor_{setting}_{iteration}.pth"
    # critic_path = f"./models_M2/BCQ_critic_{setting}_{iteration}.pth"
    # m_path = f"./models_M2/BCQ_m_{setting}_{iteration}.pth"
    # # Load the saved weights into the policy's models
    # if os.path.exists(actor_path):
    #     policy.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
    #     policy.actor.eval()  # Set to evaluation mode
    # else:
    #     print(f"Error: {actor_path} does not exist.")

    # if os.path.exists(critic_path):
    #     policy.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))
    #     policy.critic.eval()  # Set to evaluation mode
    # else:
    #     print(f"Error: {critic_path} does not exist.")

    # if os.path.exists(m_path):
    #     policy.mnetwork.load_state_dict(torch.load(m_path, map_location=torch.device('cpu')))
    #     policy.mnetwork.eval()  # Set to evaluation mode
    # else:
    #     print(f"Error: {m_path} does not exist.")

    # print('here')

    ############################# visualization #############################
    # # 创建一个 100x100 的 Q and M 值矩阵
    # num_divisions = 100
    # q_values = np.zeros((num_divisions, num_divisions))
    # m_values = np.zeros((num_divisions, num_divisions))
    # v_values = np.zeros((num_divisions, num_divisions))
    # # 定义 x 和 y 轴的范围
    # x_min, x_max = 0.0, 3.7
    # y_min, y_max = 0.0, 3.7

    # # 计算 x 和 y 轴的离散值
    # x_values = np.linspace(x_min, x_max, num_divisions)
    
    # #not only y axis need to be reversed, but also the y values
    # y_values = np.linspace(y_max, y_min, num_divisions)

    # # 用正确的 Q 值填充矩阵
    # for i, x in enumerate(x_values):
    #     for j, y in enumerate(y_values):
    #         # 创建状态张量
    #         state = torch.zeros(env.observation_space.shape[0]).to(policy.device)
    #         # 设置 x 和 y 坐标
    #         state[0] = x  # 假设第一个位置是 x 坐标
    #         state[1] = y  # 假设第二个位置是 y 坐标
    #         # 计算 Q 值并将其放置在正确的位置
    #         q_values[j, i],m_values[j, i] = compute_q_value(policy, state)
    #         v_values[j, i] = m_values[j, i] - q_values[j, i] ** 2
    #         # m_values[j, i] = compute_m_value(policy, state)
    

    # print('q_values',q_values[1,1])
    # print('m_values',m_values[1,1])
    # # 可视化 Q 值矩阵
    # plt.figure(1)
    # plt.imshow(q_values, cmap='Greens', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    # plt.colorbar(label='Q Value')
    # plt.title('Q Value Matrix')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.show()

    # plt.figure(2)
    # plt.imshow(m_values, cmap='Blues', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    # plt.colorbar(label='M Value')
    # plt.title('M Value Matrix')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.show()

    # plt.figure(3)
    # plt.imshow(v_values, cmap='Reds', interpolation='nearest', extent=[x_min, x_max, y_min, y_max])
    # plt.colorbar(label='V Value')
    # plt.title('V Value Matrix')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.show()

    ############################# policy visualization #############################
    # Visualize the policy
    # visualize_policy(policy, 'maze2d-umaze-v1', device)


    ############################# TensorBoard visualization #############################
    # data1 = np.load('results_M2/BCQ_maze2d-umaze-v1_0.npy')
    # # data2 = np.load('results/BCQ_maze2d-umaze-v1_2.npy')
    # log_dir = 'logs/'
    # writer = tf.summary.create_file_writer(log_dir)

    # # 写入第一个数据集到TensorBoard
    # with writer.as_default():
    #     for step, value in enumerate(data1):
    #         tf.summary.scalar('data1_value', value, step=step)
    #     writer.flush()  # 确保第一个数据集被写入

    # # 写入第二个数据集到TensorBoard
    # with writer.as_default():
    #     for step, value in enumerate(data2):
    #         tf.summary.scalar('data2_value', value, step=step)
    #     writer.flush()  # 确保第二个数据集被写入

    # print(f"Data written to {log_dir}")

    # data = np.load('results/BCQ_maze2d-umaze-v1_0.npy')
    # log_dir = 'logs/'
    # writer = tf.summary.create_file_writer(log_dir)

    # # 写入数据到TensorBoard
    # with writer.as_default():
    #     for step, value in enumerate(data):
    #         tf.summary.scalar('data_value', value, step=step)

    # print(f"Data written to {log_dir}")
    # critic_loss_file = 'results/Critic_Loss_maze2d-umaze-v1_0.npy'
    # m_loss_file = 'results/M_Loss_maze2d-umaze-v1_0.npy'

    # log_dir = 'logs/'
    # writer = tf.summary.create_file_writer(log_dir)

    # # 读取并写入Critic（Q网络）的损失
    # critic_loss_data = np.load(critic_loss_file)
    # with writer.as_default():
    #     for step, value in enumerate(critic_loss_data):
    #         tf.summary.scalar('Critic_Loss', value, step=step)

    # # 读取并写入M网络的损失
    # m_loss_data = np.load(m_loss_file)
    # with writer.as_default():
    #     for step, value in enumerate(m_loss_data):
    #         tf.summary.scalar('M_Loss', value, step=step)

    # print(f"Loss data written to {log_dir}")

    ############################# cropping visualization ############################# 
    # cropping('maze2d-umaze-v1', states,actions, observation_size=100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_params = {'n_channels': 3, 'n_classes': 64}
    mlp_params = {'input_dim': 640000, 'output_dim': 64}  # 根据您的实际情况调整
    feature_extractor = FeatureExtractor(unet_params, mlp_params).to(device)

    # 读取4帧图像 
    frames = []
    for i in range(4):
        cropped_image = cropping('maze2d-umaze-v1',states[i] , observation_size=100)
        cropped_image = cropped_image.copy()  # 确保没有负步长
        frames.append(cropped_image)

    # 将图像转换为PyTorch张量
    cropped_images_np = np.stack(frames)
    cropped_images_tensor = torch.from_numpy(cropped_images_np).float()
    cropped_images_tensor = cropped_images_tensor.permute(0, 3, 1, 2)  # 调整维度顺序
    cropped_images_tensor = cropped_images_tensor.to(device)

    # 使用FeatureExtractor提取特征
    feature_extractor.train()  # 确保模型处于训练模式
    mlp_output = feature_extractor(cropped_images_tensor)

    # print('mlp_output', mlp_output.shape)


