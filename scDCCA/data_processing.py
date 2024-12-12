import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(data, batch_size=256):
    """
    加载数据并创建 DataLoader
    :param data: 输入数据
    :param batch_size: 批量大小
    :return: DataLoader
    """
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def preprocess_data(raw_data):
    """
    数据预处理函数
    :param raw_data: 原始数据
    :return: 预处理后的数据
    """
    # 这里可以添加任何预处理逻辑，比如归一化、去除异常值等
    processed_data = (raw_data - raw_data.mean()) / raw_data.std()  # 示例归一化
    return processed_data
