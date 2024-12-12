import torch
from torch.utils.data import DataLoader, TensorDataset
from basic_deep_learning_model import MyModel

# 测试用例

def test_mymodel():
    # 模型参数
    input_dim = 2000  # 输入维度
    z_dim = 32        # 潜在维度
    n_clusters = 10   # 聚类数量

    # 创建模型
    model = MyModel(input_dim, z_dim, n_clusters)

    # 创建虚拟数据
    X = torch.rand(100, input_dim)  # 100个样本

    # 创建数据加载器
    dataset = TensorDataset(X)  # 只传递输入数据
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    model.fit(data_loader, num_epochs=5, learning_rate=0.001)

if __name__ == '__main__':
    test_mymodel()
