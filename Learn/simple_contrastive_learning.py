# 简单的对比学习示例
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成简单的合成数据
def generate_data(n_samples=100):
    """
    生成两组相关的数据：
    - 第一组是2D平面上的点
    - 第二组是第一组加上随机噪声
    """
    # 生成第一组数据：在2D平面上的两个高斯分布
    x1 = np.random.normal(0, 1, (n_samples//2, 2))
    x2 = np.random.normal(3, 1, (n_samples//2, 2))
    x = np.vstack([x1, x2])
    
    # 生成第二组数据：添加噪声
    noise = np.random.normal(0, 0.1, x.shape)
    x_noisy = x + noise
    
    return torch.FloatTensor(x), torch.FloatTensor(x_noisy)

# 2. 定义一个简单的编码器网络
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# 3. 定义对比损失
class SimpleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        z1, z2: 两组编码后的数据 [batch_size, dim]
        """
        # 归一化编码
        z1_norm = torch.nn.functional.normalize(z1, dim=1)
        z2_norm = torch.nn.functional.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        
        # 对角线上的元素应该最大（正样本对）
        labels = torch.arange(z1.shape[0]).to(z1.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return loss

# 4. 训练函数
def train_contrastive(encoder, data1, data2, n_epochs=100):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    contrastive_loss = SimpleContrastiveLoss()
    
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 获取两组数据的编码
        z1 = encoder(data1)
        z2 = encoder(data2)
        
        # 计算损失
        loss = contrastive_loss(z1, z2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 5. 主函数
def main():
    # 生成数据
    data1, data2 = generate_data()
    
    # 创建编码器
    encoder = SimpleEncoder()
    
    # 训练模型
    losses = train_contrastive(encoder, data1, data2)
    
    # 可视化训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # 可视化编码结果
    with torch.no_grad():
        z1 = encoder(data1)
        z2 = encoder(data2)
    
    plt.figure(figsize=(10, 5))
    
    # 原始数据
    plt.subplot(121)
    plt.scatter(data1[:, 0], data1[:, 1], c='blue', alpha=0.5, label='原始数据')
    plt.scatter(data2[:, 0], data2[:, 1], c='red', alpha=0.5, label='加噪声数据')
    plt.title('原始数据分布')
    plt.legend()
    
    # 编码后的数据
    plt.subplot(122)
    plt.scatter(z1[:, 0], z1[:, 1], c='blue', alpha=0.5, label='编码1')
    plt.scatter(z2[:, 0], z2[:, 1], c='red', alpha=0.5, label='编码2')
    plt.title('编码后的数据分布')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
