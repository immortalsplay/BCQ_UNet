import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.up1 = DoubleConv(256, 128)
        self.up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x3 = self.down1(x2)
        x4 = F.max_pool2d(x3, 2)
        x5 = self.down2(x4)
        x = F.interpolate(x5, scale_factor=2, mode='nearest')
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up2(x)
        logits = self.outc(x)
        return logits

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout_rate=0.5):
        super(MLP, self).__init__()
        
        # 创建模型的层次结构
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # 第一层接受输入维度
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                # 后续层接受上一层的输出作为输入
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))

            # 批量归一化层
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout层
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 把所有层次结构组合起来
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        return self.layers(x)

class FeatureExtractor(nn.Module):
    def __init__(self, unet_params, mlp_params):
        super(FeatureExtractor, self).__init__()
        self.unet = UNet(**unet_params)
        self.mlp = MLP(**mlp_params)

    def forward(self, x):
        features = self.unet(x)
        flattened_features = features.reshape(features.size(0), -1)
        return self.mlp(flattened_features)

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

def cropping(env, state ,observation_size = 190):

    if hasattr(env, 'viewer') and env.viewer is not None:
        # 设置俯视角度
        env.viewer.cam.elevation = -90  # 俯视角度
        env.viewer.cam.azimuth = 90      # 可选，更改方位角

    # set state
    # print('state',state)
    qpos = state[:2]
    qvel = state[2:]
    # qpos = np.array([state[0], state[1]])
    # qvel = np.array([state[2], state[3]])
    env.set_state(qpos, qvel)

    img = env.render(mode='rgb_array')

    # agent transform_coordinates
    plot_x, plot_y = transform_coordinates(
        qpos,
        env_bottom_left=(-0.107, 0.13),
        env_top_right=(3.72, 3.72),
        plot_bottom_left=(156, 345),
        plot_top_right=(345, 156)
    )
    # print('state[:2]',state[:2])
    # print('plot_x, plot_y',plot_x, plot_y)

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
# # 假设你的输入特征维度是1024，输出特征维度是你BCQ模型所需的大小
# input_dim = 1024
# output_dim = 64  # 这里只是一个示例值，你需要根据你的BCQ模型来设定

# # 实例化MLP模型
# mlp_model = MLP(input_dim=input_dim, output_dim=output_dim)

# # 检查模型结构
# print(mlp_model)
