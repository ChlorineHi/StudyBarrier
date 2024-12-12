import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class DeepClusteringModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_clusters):
        """
        深度聚类模型
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dims: 隐藏层维度列表
        - n_clusters: 聚类数量
        """
        super(DeepClusteringModel, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # 聚类层
        self.encoder = nn.Sequential(*encoder_layers)
        self.z_dim = hidden_dims[-1]
        
        # 聚类中心
        # 生成聚类中心
        # 可以在反向传播的时候进行数据的更新
        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, self.z_dim), 
            requires_grad=True
        )
        
        # 解码器（对称的）
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.n_clusters = n_clusters
    
    def forward(self , x):
        z = self.encode(x)
        x_recon = self.decode(z)
        q = self.soft_assign(z)

        return z , x_recon , q

    def encode(self, x):
        """编码"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码"""
        return self.decoder(z)
    
    def soft_assign(self, z):
        """
        软分配：计算样本到聚类中心的概率
        使用学生t分布（类似于t-SNE）
        """
        # alpha = 1.0
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2) / 1.0)
        q = q**2 / q.sum(0)
        return (q.t() / q.sum(1)).t()

def target_distribution(q):
    """
    生成目标分布，使软分配更加尖锐
    """
    p = q**2 / q.sum(0)
    return (p.t() / p.sum(1)).t()

def train_deep_clustering(X, true_labels, n_clusters=3, epochs=100, lr=0.001):
    """
    深度聚类训练流程
    
    参数:
    - X: 输入特征矩阵
    - true_labels: 真实标签（用于评估）
    - n_clusters: 聚类数量
    - epochs: 训练轮数
    - lr: 学习率
    """
    # 数据预处理
    X = StandardScaler().fit_transform(X)
    X = torch.FloatTensor(X)
    
    # 模型初始化
    input_dim = X.shape[1]
    model = DeepClusteringModel(
        input_dim=input_dim, 
        hidden_dims=[64, 32], 
        n_clusters=n_clusters
    )
    
    # 优化器
    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': [model.cluster_centers], 'lr': lr * 10}
    ], lr=lr)
    
    # 损失函数
    reconstruction_loss_fn = nn.MSELoss()
    
    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        z = model.encode(X)
        x_recon = model.decode(z)
        
        # 重构损失
        recon_loss = reconstruction_loss_fn(x_recon, X)
        
        # 软分配
        q = model.soft_assign(z)
        p = target_distribution(q)
        
        # 聚类损失（KL散度）
        cluster_loss = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        
        # 总损失
        loss = recon_loss + cluster_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每10轮评估一次
        if epoch % 10 == 0:
            # 预测标签
            pred_labels = torch.argmax(q, dim=1).numpy()
            
            # 评估指标
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}, '
                  f'ARI = {ari:.4f}, NMI = {nmi:.4f}')
    
    return model, pred_labels

def main():
    # 生成示例数据
    X, true_labels = make_blobs(
        n_samples=300, 
        centers=3, 
        cluster_std=0.6, 
        random_state=42
    )
    
    # 训练深度聚类模型
    model, pred_labels = train_deep_clustering(X, true_labels)
    
    # 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
        plt.title('True Labels')
        
        plt.subplot(132)
        plt.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis')
        plt.title('Predicted Labels')
        
        plt.subplot(133)
        z = model.encode(torch.FloatTensor(X))
        plt.scatter(z.detach().numpy()[:, 0], z.detach().numpy()[:, 1], c=pred_labels, cmap='viridis')
        plt.title('Latent Space')
        
        plt.tight_layout()
        plt.savefig('E:\\Study\\scDCCA-main\\Learn\\clustering_results.png')
        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == '__main__':
    main()
