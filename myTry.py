import scanpy as sc
import pandas as pd
import numpy as np
# 创建 AnnData 对象
adata = sc.AnnData(X=np.random.rand(100, 20))
print(adata)
# 添加 DCA_split 列
adata.obs['DCA_split'] = ['train', 'test'] * 50

# 转换为类别类型
adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')

# 查看类型
print(adata.obs['DCA_split'].dtype)  # category

# 打印信息
print('Preprocessed {} genes and {} cells'.format(adata.n_vars, adata.n_obs))