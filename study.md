生物信息学参数总览
1. 数据预处理参数
input_dim：输入特征维度
n_features：特征数量
n_cells：细胞数量
normalization_method：标准化方法（如 log-normalization, z-score）
2. 降维与表示学习参数
z_dim：潜在空间维度
latent_dim：隐藏层维度
n_components：主成分分析（PCA）的成分数
perplexity：t-SNE算法的困惑度
3. 聚类相关参数
n_clusters：聚类数量
cluster_method：聚类算法（K-means, Louvain, Leiden）
resolution：社区检测分辨率
min_cluster_size：最小聚类大小
4. 深度学习模型参数
learning_rate：学习率
batch_size：批次大小
epochs：训练轮数
dropout_rate：随机失活率
activation：激活函数类型
5. 正则化参数
lambda：正则化强度
alpha：L1正则化系数
beta：L2正则化系数
6. 损失函数相关
margin：对比损失边界
temperature：对比学习温度参数
gamma：损失权重
7. 细胞类型鉴定参数
marker_genes：标志基因列表
confidence_threshold：置信度阈值
max_markers：最大标志基因数
8. 批次效应校正
batch_correction_method：批次效应校正算法
reference_batch：参考批次
9. 特征选择参数
n_top_genes：顶部基因数量
min_mean：最小平均表达
max_mean：最大平均表达
min_dispersion：最小分散度
10. 拟时序分析参数
root_cell：起始细胞
branch_method：分支推断方法
pseudotime_method：拟时序算法
11. 数据增强参数
augmentation_ratio：数据增强比例
noise_level：噪声水平
12. 模型正则化
l1_reg：L1正则化
l2_reg：L2正则化
early_stopping：早停
weight_decay：权重衰减


输入(2000维) 
-> 编码器(2000->256->64) 
-> 均值层(64->32) 
-> 解码器(32->64->256) 
-> 均值层(256->2000)
   离散度层(256->2000)
   概率层(256->2000)
-> 重建输出(2000维)

dim=0：沿着第0维（行）进行归一化
dim=1：沿着第1维（列）进行归一化

数据去噪
通过计算均值抑制极端值
平滑数据中的随机波动
提取稳定的特征表示
聚类辅助
为聚类算法提供更紧凑的特征表示
将相似数据点聚集在一起
降低特征空间的复杂性

对比学习的表示顺序
完全正确！这是一个非常精准的总结。这种学习策略可以形象地描述为"由微观到宏观"的特征学习过程：

单细胞级别（微观）：InstanceLoss
目标：区分不同的单个细胞
让同一个细胞的不同视角特征相似
让不同细胞的特征有明显区别
类似于：先确保每个细胞都有自己独特的"指纹"
群体级别（宏观）：ClusterLoss
目标：区分不同的细胞群体/类型
让同一类型的细胞特征相似
让不同类型的细胞群体特征有明显区别
类似于：再将有相似"指纹"的细胞归类