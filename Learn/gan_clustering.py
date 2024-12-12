import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        """
        对比损失函数
        x1, x2: 两个样本的表示
        label: 标签，1表示同类，0表示异类
        """
        euclidean_distance = torch.nn.functional.pairwise_distance(x1, x2)
        
        # 同类损失：拉近距离
        same_class_loss = label * torch.pow(euclidean_distance, 2)
        
        # 异类损失：推远距离
        different_class_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        return torch.mean(same_class_loss + different_class_loss)

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ContrastiveEncoder, self).__init__()
        
        # 编码器网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = hidden_dim
        
        # 最终输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim, n_clusters):
        super(Generator, self).__init__()
        
        # 生成器网络
        layers = []
        prev_dim = z_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = hidden_dim
        
        # 最终输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # 聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, output_dim), 
            requires_grad=True
        )
        
        self.n_clusters = n_clusters

    def forward(self, z, cluster_idx=None):
        # 如果指定了簇索引，则将簇中心添加到输入噪声中
        if cluster_idx is not None:
            z = z + self.cluster_centers[cluster_idx]
        
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_clusters):
        super(Discriminator, self).__init__()
        
        # 保存簇数量
        self.n_clusters = n_clusters
        
        # 判别器网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = hidden_dim
        
        # 网络主体
        self.model = nn.Sequential(*layers)
        
        # 分开的输出层
        self.validity_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        self.cluster_layer = nn.Linear(prev_dim, n_clusters)

    def forward(self, x):
        # 网络主体
        h = self.model(x)
        
        # 分别计算真/假判断和簇分类
        validity = self.validity_layer(h)
        cluster_pred = self.cluster_layer(h)
        
        return validity.squeeze(), cluster_pred

class DataAugmentation:
    def __init__(self, noise_level=0.1, dropout_rate=0.1):
        """
        数据增强类
        
        参数:
        - noise_level: 高斯噪声强度
        - dropout_rate: 特征丢弃比例
        """
        self.noise_level = noise_level
        self.dropout_rate = dropout_rate
    
    def __call__(self, x):
        """
        数据增强方法
        
        参数:
        - x: 输入张量 (batch_size, feature_dim)
        
        返回:
        - 增强后的张量
        """
        # 克隆输入以避免原地修改
        augmented_x = x.clone()
        
        # 高斯噪声
        noise = torch.randn_like(augmented_x) * self.noise_level
        augmented_x += noise
        
        # 特征丢弃
        mask = torch.rand_like(augmented_x) > self.dropout_rate
        augmented_x *= mask
        
        # 随机线性变换
        scale = torch.rand(augmented_x.shape[0], 1) * 0.5 + 0.75  # 0.75 到 1.25 之间
        shift = torch.randn(augmented_x.shape[0], 1) * 0.1
        augmented_x = augmented_x * scale + shift
        
        return augmented_x

def soft_assign(z, cluster_centers, alpha=1.0):
    """
    软分配：计算样本到聚类中心的概率
    
    Args:
    - z: 输入样本张量 (batch_size, feature_dim)
    - cluster_centers: 聚类中心张量 (n_clusters, feature_dim)
    - alpha: 软分配的温度参数
    
    Returns:
    - 软分配概率 (batch_size, n_clusters)
    """
    # 确保 z 和 cluster_centers 是 FloatTensor
    z = z.float()
    cluster_centers = cluster_centers.float()
    
    # 计算样本到每个聚类中心的距离
    # z: (batch_size, 1, feature_dim)
    # cluster_centers: (1, n_clusters, feature_dim)
    z_expanded = z.unsqueeze(1)  # 扩展 z 的维度
    centers_expanded = cluster_centers.unsqueeze(0)  # 扩展聚类中心的维度
    
    # 计算欧氏距离的平方
    dist_sq = torch.sum((z_expanded - centers_expanded)**2, dim=-1)
    
    # 软分配：使用 t 分布（类似 t-SNE）
    q = 1.0 / (1.0 + dist_sq / alpha)
    
    # 归一化，使每个样本的分配概率和为1
    q = q / torch.sum(q, dim=1, keepdim=True)
    
    return q

def train_gan_clustering(X, true_labels, n_clusters=3, epochs=200, lr=0.0002, batch_size=64):
    # 数据预处理
    X = StandardScaler().fit_transform(X)
    X = torch.FloatTensor(X)
    
    # 数据增强
    data_augmentation = DataAugmentation(
        noise_level=0.1,  # 噪声强度
        dropout_rate=0.1  # 特征丢弃率
    )
    
    # 获取输入维度
    input_dim = X.shape[1]
    z_dim = 64  # 固定隐空间维度
    contrastive_dim = 32  # 对比学习表示维度

    # 生成初始噪声
    def generate_noise(batch_size):
        return torch.randn(batch_size, z_dim)

    # 初始化模型
    generator = Generator(
        z_dim=z_dim, 
        hidden_dims=[64, 128], 
        output_dim=input_dim, 
        n_clusters=n_clusters
    )
    
    discriminator = Discriminator(
        input_dim=input_dim, 
        hidden_dims=[128, 64],
        n_clusters=n_clusters
    )

    # 对比学习编码器
    contrastive_encoder = ContrastiveEncoder(
        input_dim=input_dim, 
        hidden_dims=[64, 128], 
        output_dim=contrastive_dim
    )

    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    c_optimizer = optim.Adam(contrastive_encoder.parameters(), lr=lr)

    # 损失函数
    adversarial_loss = nn.BCELoss()
    cluster_loss = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(margin=1.0)

    # 训练循环
    for epoch in range(epochs):
        # 随机选择批次
        indices = torch.randperm(len(X))[:batch_size]
        real_samples = X[indices]
        
        # 数据增强
        augmented_samples = data_augmentation(real_samples)
        
        # 生成噪声
        z_noise = generate_noise(batch_size)
        
        # 生成假样本
        fake_samples = generator(z_noise)
        
        # 对比学习
        c_optimizer.zero_grad()
        
        # 对比学习样本对
        anchor_samples = real_samples[:batch_size//2]
        positive_samples = augmented_samples[:batch_size//2]
        
        # 生成标签
        labels = torch.ones(batch_size//2)
        
        # 编码
        anchor_embeddings = contrastive_encoder(anchor_samples)
        positive_embeddings = contrastive_encoder(positive_samples)
        
        # 对比损失
        contrastive_loss = contrastive_criterion(
            anchor_embeddings, 
            positive_embeddings, 
            labels
        )
        contrastive_loss.backward()
        c_optimizer.step()
        
        # 判别器训练
        d_optimizer.zero_grad()
        
        # 真实样本
        real_validity, real_cluster_pred = discriminator(real_samples)
        d_real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
        
        # 生成样本
        fake_validity, _ = discriminator(fake_samples.detach())
        d_fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
        
        # 簇分类损失
        d_cluster_loss = cluster_loss(
            real_cluster_pred, 
            torch.tensor(true_labels[indices], dtype=torch.long)
        )
        
        # 判别器总损失
        d_loss = d_real_loss + d_fake_loss + d_cluster_loss
        d_loss.backward()
        d_optimizer.step()
        
        # 生成器训练
        g_optimizer.zero_grad()
        
        # 生成器对抗损失
        fake_validity, _ = discriminator(fake_samples)
        g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        
        # 软分配损失
        z_noise_assigned = soft_assign(fake_samples, generator.cluster_centers)
        g_cluster_loss = torch.mean(torch.sum(z_noise_assigned * torch.log(z_noise_assigned + 1e-10), dim=1))
        
        # 生成器总损失
        g_total_loss = g_loss + g_cluster_loss
        g_total_loss.backward()
        g_optimizer.step()
        
        # 每20轮评估
        if epoch % 20 == 0:
            # 预测标签
            with torch.no_grad():
                z = soft_assign(torch.FloatTensor(X), generator.cluster_centers)
                pred_labels = torch.argmax(z, dim=1).numpy()
            
            # 评估指标
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            print(f'Epoch {epoch}: '
                  f'D Loss = {d_loss.item():.4f}, '
                  f'G Loss = {g_total_loss.item():.4f}, '
                  f'Contrastive Loss = {contrastive_loss.item():.4f}, '
                  f'ARI = {ari:.4f}, NMI = {nmi:.4f}')
    
    return generator, contrastive_encoder, pred_labels

def main():
    # 生成示例数据
    X, true_labels = make_blobs(
        n_samples=300, 
        centers=3, 
        cluster_std=0.6, 
        random_state=42
    )
    
    # 训练GAN聚类模型
    model, contrastive_encoder, pred_labels = train_gan_clustering(X, true_labels)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 生成样本
    z_noise = torch.randn(300, 64)  # 使用与训练时相同的噪声维度
    generated_samples = model(z_noise).detach().numpy()
    
    # 原始数据可视化
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
    plt.title('Original Data')
    
    # 生成数据可视化
    plt.subplot(132)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c=pred_labels, cmap='viridis')
    plt.title('Generated Data')
    
    # 聚类结果可视化
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis')
    plt.title('Clustering Results')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
