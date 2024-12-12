# StudyBarrier

## 项目简介
StudyBarrier 是一个深度学习框架，旨在处理高维数据并实现聚类分析。该项目提供了数据预处理、模型训练和评估等功能，适用于生物信息学、图像处理等领域。

## 主要特性
- **数据预处理**：支持数据加载、归一化和批处理。
- **深度学习模型**：实现了编码器-解码器结构，支持自编码器和聚类功能。
- **训练与评估**：提供训练和评估模型的功能，支持多种损失函数。

## 安装
1. 克隆该仓库：
   ```bash
   git clone https://github.com/ChlorineHi/StudyBarrier.git
   cd StudyBarrier
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用
1. 数据预处理：使用 `data_processing.py` 中的函数加载和处理数据。
2. 模型训练：使用 `mymodel_new.py` 中的 `MyModel` 类进行模型训练。
3. 评估模型：使用评估函数计算模型性能指标。

## 示例
```python
from data_processing import load_data, preprocess_data
from mymodel_new import MyModel

# 加载和预处理数据
raw_data = ...  # 加载原始数据
processed_data = preprocess_data(raw_data)

# 创建模型
model = MyModel(input_dim=2000, z_dim=32, n_clusters=10)

# 训练模型
model.fit(processed_data)
```

## 贡献
欢迎贡献！请提交 Pull Request 或者 Issue。

## 许可证
本项目采用 MIT 许可证，详细信息请参见 LICENSE 文件。
