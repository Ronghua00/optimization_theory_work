import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# 简单的卷积神经网络用于深度估计
class DepthEstimationCNN(nn.Module):
    def __init__(self):
        super(DepthEstimationCNN, self).__init__()
        # 卷积层提取特征
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 转换到深度图尺寸
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)  # 输出深度图，大小为 [batch_size, 1, 32, 32]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # 输出深度图，形状为 [batch_size, 1, 32, 32]
        return x


# 生成一个简单的模拟数据集（仅用于演示）
class SimpleDepthDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.images = torch.rand(num_samples, 3, 32, 32)  # 随机生成图像
        self.depths = torch.rand(num_samples, 1, 32, 32)  # 随机生成深度图

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.depths[idx]

# L2损失函数（重建误差）
def l2_loss(pred, target):
    return torch.mean((pred - target) ** 2)

# 平滑正则化损失函数
def smoothness_loss(pred_depth):
    # 计算梯度
    depth_grad_x = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])  # x方向梯度
    depth_grad_y = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])  # y方向梯度

    # 如果需要，裁剪梯度张量，确保与输入深度图尺寸一致
    depth_grad_x = depth_grad_x[:, :, :-1, :]  # 去掉多余的列
    depth_grad_y = depth_grad_y[:, :, :, :-1]  # 去掉多余的行

    # 返回平滑损失
    return torch.mean(depth_grad_x + depth_grad_y)

# 总损失函数
def total_loss(pred_depth, true_depth, lambda_smooth=0.1):
    # L2损失
    loss_reconstruction = l2_loss(pred_depth, true_depth)
    # 平滑损失
    loss_smooth = smoothness_loss(pred_depth)
    # 总损失
    return loss_reconstruction + lambda_smooth * loss_smooth

# 训练过程
def train(model, dataloader, optimizer, num_epochs=10):
    model.train()  # 设置为训练模式
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, depths in dataloader:
            optimizer.zero_grad()  # 清空梯度
            pred_depths = model(images)  # 前向传播
            loss = total_loss(pred_depths, depths)  # 计算总损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# 测试过程（仅用于查看输出）
def test(model, dataloader):
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for images, depths in dataloader:
            pred_depths = model(images)
            # 显示结果
            plt.subplot(1, 2, 1)
            plt.imshow(depths[0].cpu().numpy().squeeze(), cmap='jet')
            plt.title("True Depth")
            plt.subplot(1, 2, 2)
            plt.imshow(pred_depths[0].cpu().numpy().squeeze(), cmap='jet')
            plt.title("Predicted Depth")
            plt.show()

# 主程序
if __name__ == '__main__':
    # 生成简单的模拟数据集
    dataset = SimpleDepthDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型、优化器
    model = DepthEstimationCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    train(model, dataloader, optimizer, num_epochs=10)

    # 测试模型
    test(model, dataloader)
