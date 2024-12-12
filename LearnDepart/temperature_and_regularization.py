class MyModel(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], 
                 activation="relu", sigma=1.0, alpha=1.0, gamma=1.0, i_temp=0.5, c_temp=1.0, i_reg=0.5, c_reg=0.2, feature_dim=32, device='cpu'):
        super(MyModel, self).__init__()
        self.i_temp = i_temp  # 对比学习温度
        self.c_temp = c_temp  # 聚类学习温度
        self.i_reg = i_reg  # 对比学习系数
        self.c_reg = c_reg  # 聚类学习系数

    # 其他方法...
