import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
import pandas as pd
import scanpy as sp
from preprocess import  * 
####用于导入指定模块中的全部定义。
from collections import defaultdict
from sklearn import preprocessing
import random
from contrastive_loss import ClusterLoss, InstanceLoss
from pandas import Series
from time import time as get_time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

import os
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    normalized_mutual_info_score, 
    adjusted_rand_score, 
    homogeneity_score, 
    completeness_score
)

def evaluate(y_true, y_pred):
    """
    计算聚类性能指标
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    
    返回:
    准确率、F1分数、NMI、ARI、同质性、完整性
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        homo = homogeneity_score(y_true, y_pred)
        comp = completeness_score(y_true, y_pred)
        
        return acc, f1, nmi, ari, homo, comp
    except Exception as e:
        logging.error(f"Error in evaluate function: {e}")
        return 0, 0, 0, 0, 0, 0

def show_info():
    #计算消耗内存
    pid = os.getpid()
    # 模块名比较容易理解：获得当前进程的pid
    p = psutil.Process(pid)
    # 根据pid找到进程，进而找到占用的内存值
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    return memory

'''
添加层之间的数据函数
'''
def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class MyModel(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1.0, alpha=1.0, gamma=1.0, i_temp=0.5, c_temp=1.0, i_reg=0.5, c_reg=0.2, feature_dim=32, device='cpu'):
        super(MyModel, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = buildNetwork([input_dim]+encodeLayer, activation=activation)###2000-256-64
        self.decoder = buildNetwork([z_dim]+decodeLayer, activation=activation)###32-64-256
        self.decoder1 = buildNetwork([z_dim]+decodeLayer, activation=activation)###32-64-256
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)###64-32
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())##256-2000
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())##256-2000
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())##256-2000
        
        self.i_temp = i_temp
        self.c_temp = c_temp
        self.i_reg = i_reg####0.5
        self.c_reg = c_reg####0.2

        self.instance_projector = nn.Sequential(####实例投影
            nn.Linear(z_dim, z_dim),###32-32
            nn.ReLU(),
            nn.Linear(z_dim, feature_dim))###32-2000
        
        self.cluster_projector = nn.Sequential(#####聚类投影，z_dim维投影到n_clusters维
            nn.Linear(z_dim, n_clusters),####32-10
            nn.Softmax(dim=1))

        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))###生成一个10*32的质心坐标，即每个簇的坐标都是32维
        self.zinb_loss = ZINBLoss()
        self.to(device)
    

    '''
    软分配的方法 深度聚类
    允许数据点"部分"属于多个聚类
    捕捉数据的复杂分布特征
    在单细胞分析中特别有用,因为生物学细胞状态often是连续的
    这是一个软分配(soft assignment)方法，用于计算数据点属于不同聚类中心的概率。
    这是深度聚类(Deep Clustering)中常用的技术,特别是在学生t分布邻域嵌入(t-SNE)的思想基础上。
    '''
    def soft_assign(self, z):####算z所属的软标签，即用学生分布度量z与self.mu的相似度
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)###对于每一个cell都有一个10维度的q,因为属于每个簇的概率不同
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q  ###对于每一个cell都有一个10维度的q,因为属于每个簇的概率不同，所以q是268*10的矩阵
    
    '''
    是对软分配结果的进一步处理。
    放大高置信度的聚类概率
    抑制低置信度的聚类概率
    生成一个更"尖锐"的分布
    '''
    def target_distribution(self, q):###
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    # 随机丢弃
    def x_drop(self, x, p=0.2):
        # shape[0]样本数(行) shape[1]特征数 
        mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
        mask = torch.vstack(mask_list)
        new_x = x.clone()
        new_x[mask] = 0.0
        return new_x
    
    def forward(self, x):###作用是对数据进行预训练，得到相应的中间变量，z0, q是用原始的x得到的，_mean, _disp, _pi是用加了噪声的x得到的
        
        '''
        Xu : 
        '''
        h0 = self.encoder(x)
        
        # 添加随机噪声，用于正则化
        z = self._enc_mu(h0)####64-32
        # 带噪声的编码
        h = self.decoder(z)####32-64-256
        
        h1 = self.decoder1(z)####32-64-256
        h = (h1 + h) / 2###求平均是为了减小波动
        
        # 重构参数
        '''
        计算负二项分布的三个参数
        用于数据重构和损失计算
        '''
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        # 原始数据编码和软分配
        '''
        直接对原始数据进行编码
        计算软分配（聚类概率）
        返回值：
        z0:原始数据的潜在表示
        q:软分配结果
        _mean, _disp, _pi重构参数
        '''
        h0 = self.encoder(x)##将输入直接进行编码，2000-256-64
        z0 = self._enc_mu(h0)####64-32
        q = self.soft_assign(z0)###直接对隐藏层z0求软分配，度量的是z0和self.mu的相似性
        return z0, q, _mean, _disp, _pi
    
    
    def calc_ssl_lossv1(self, x1, x2):
        z1, _, _, _, _ = self.forward(x1)###将x1送到编码层2000-256-64-32得到z1
        z2, _, _, _, _ = self.forward(x2)##将x2送到编码层2000-256-64-32得到z2
        
        instance_loss = InstanceLoss(x1.shape[0], self.i_temp)###x1.shape[0]=256,self.i_temp=0.5,算instance_loss
        return instance_loss.forward(z1 ,z2)###返回instance_loss
    
    
    def calc_ssl_lossv2(self, x1, x2):###计算对比过程的聚类损失，2000维
        # _, q1, _, _, _ = self.forward(x1)
        # _, q2, _, _, _ = self.forward(x2)
        # cluster_loss = ClusterLoss(self.n_clusters, self.c_temp)
        # return cluster_loss.forward(q1, q2)
        z1, _, _, _, _ = self.forward(x1)
        z2, _, _, _, _ = self.forward(x2)###32维
        c1 = self.cluster_projector(z1)
        c2 = self.cluster_projector(z2)#####10维
        cluster_loss = ClusterLoss(self.n_clusters, self.c_temp)###定义cluster_loss的计算用ClusterLoss函数，具体去看这个函数
        return cluster_loss.forward(c1, c2)
    
    """
    def calc_ssl_loss(self, x, p=0.2):
        x1 = self.x_drop(x, p)
        x2 = self.x_drop(x, p)
        z1, _, _, _, _ = self.forward(x1)
        z2, _, _, _, _ = self.forward(x2)
        z1 = self.instance_projector(z1)
        z2 = self.instance_projector(z2)
        ssl_loss1 = InstanceLoss(x.shape[0], self.i_temp)
        instance_loss = ssl_loss1.forward(z1, z2)
        c1 = self.cluster_projector(z1)
        c2 = self.cluster_projector(z2)
        ssl_loss2 = ClusterLoss(self.n_clusters, self.c_temp)
        cluster_loss = ssl_loss2.forward(z1, z2)
        return instance_loss, cluster_loss
    """
    
    def encodeBatch(self, X, batch_size=256):
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # 确保 X 是 numpy 数组
        if torch.is_tensor(X):
            X = X.cpu().numpy()

        # 处理 NaN 值
        # 使用均值填充 NaN
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        use_cuda = torch.cuda.is_available()
        
        # 将数据转换为 PyTorch 张量
        X = torch.tensor(X).float()
        if use_cuda:
            X = X.cuda()

        self.eval()
        # 将数据分批编码
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*num/batch_size))
        
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            
            with torch.no_grad():
                z, _, _, _, _ = self.forward(xbatch)
                encoded.append(z)
        
        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))####这里求了均值，所以后面会* len(inputs)
        kldloss = kld(p, q)
        return self.gamma*kldloss ####self.gamma=1.0

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        import os
        import torch
        import torch.optim as optim
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建 model 目录
        os.makedirs('model_sample', exist_ok=True)

        # 记录超参数
        with open('model_sample/pretrain_hyperparameters.txt', 'w') as f:
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Sigma: {self.sigma}\n")
            f.write(f"Alpha: {self.alpha}\n")
            f.write(f"Gamma: {self.gamma}\n")

        # 初始化记录指标的列表
        train_losses = []
        zinb_losses = []
        kl_losses = []

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        # 定义优化器
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Pretraining stage")
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_zinb_loss = 0.0
            epoch_kl_loss = 0.0

            for batch_x, batch_x_raw, batch_sf in dataloader:
                if use_cuda:
                    batch_x = batch_x.cuda()
                    batch_x_raw = batch_x_raw.cuda()
                    batch_sf = batch_sf.cuda()

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                z0, q, _mean, _disp, _pi = self.forward(batch_x)

                # 计算 ZINB 重构损失
                zinb_loss = self.zinb_loss(batch_x_raw, _mean, _disp, _pi, batch_sf)

                # 计算 KL 散度损失
                p = self.target_distribution(q)
                kl_loss = self.cluster_loss(p, q)

                # 总损失
                loss = zinb_loss + kl_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                epoch_zinb_loss += zinb_loss.item()
                epoch_kl_loss += kl_loss.item()

            # 每个 epoch 的平均损失
            avg_loss = epoch_loss / len(dataloader)
            avg_zinb_loss = epoch_zinb_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)

            # 记录损失
            train_losses.append(avg_loss)
            zinb_losses.append(avg_zinb_loss)
            kl_losses.append(avg_kl_loss)

            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
                      f'ZINB Loss: {avg_zinb_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

        # 绘制损失图
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.plot(train_losses, label='Total Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(132)
        plt.plot(zinb_losses, label='ZINB Loss', color='green')
        plt.title('ZINB Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(133)
        plt.plot(kl_losses, label='KL Loss', color='red')
        plt.title('KL Divergence Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('model_sample/pretrain_losses.png')
        plt.close()

        # 保存损失数据为 CSV
        loss_data = np.column_stack((train_losses, zinb_losses, kl_losses))
        np.savetxt('model_sample/pretrain_losses.csv', loss_data, delimiter=',', 
                   header='Total_Loss,ZINB_Loss,KL_Loss', comments='')

        # 保存模型
        if ae_save:
            torch.save(self.state_dict(), ae_weights)

        return train_losses, zinb_losses, kl_losses

    def fit(self, X, X_raw, sf, y, lr=1.0, batch_size=256, num_epochs=100, save_path=None):
        try:
            # 训练前的准备工作
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            # 数据转换
            X = torch.FloatTensor(X).to(self.device)
            X_raw = torch.FloatTensor(X_raw).to(self.device)
            sf = torch.FloatTensor(sf).to(self.device)

            # 数据加载
            train_dataset = torch.utils.data.TensorDataset(X, X_raw, sf)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # 优化器和损失函数
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
            # 损失记录
            total_losses = []
            cluster_losses = []
            recon_losses = []
            ssl_losses = []

            # 训练过程
            for epoch in range(num_epochs):
                self.train()
                epoch_total_loss = 0
                epoch_cluster_loss = 0
                epoch_recon_loss = 0
                epoch_ssl_loss = 0

                for batch_x, batch_x_raw, batch_sf in train_loader:
                    optimizer.zero_grad()

                    # 前向传播
                    z, z_c, recon_x, q = self(batch_x, batch_sf)

                    # 损失计算
                    recon_loss = self.loss_fn(batch_x_raw, recon_x, batch_sf)
                    cluster_loss = self.cluster_loss(z_c)
                    ssl_loss = self.instance_loss(z)
                    total_loss = recon_loss + self.alpha * cluster_loss + self.gamma * ssl_loss

                    # 反向传播
                    total_loss.backward()
                    optimizer.step()

                    # 记录损失
                    epoch_total_loss += total_loss.item()
                    epoch_cluster_loss += cluster_loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_ssl_loss += ssl_loss.item()

                # 平均损失
                total_losses.append(epoch_total_loss / len(train_loader))
                cluster_losses.append(epoch_cluster_loss / len(train_loader))
                recon_losses.append(epoch_recon_loss / len(train_loader))
                ssl_losses.append(epoch_ssl_loss / len(train_loader))

                # 打印损失
                if epoch % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Total Loss: {total_losses[-1]:.4f}, '
                          f'Cluster Loss: {cluster_losses[-1]:.4f}, '
                          f'Recon Loss: {recon_losses[-1]:.4f}, '
                          f'SSL Loss: {ssl_losses[-1]:.4f}')

            # 预测聚类标签
            with torch.no_grad():
                _, z_c, _, _ = self(X, sf)
                self.y_pred = self.kmeans.predict(z_c.cpu().numpy())

            # 保存模型
            if save_path:
                torch.save(self.state_dict(), save_path)

            return total_losses, cluster_losses, recon_losses, ssl_losses

        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # 添加详细的调试输出
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    time_start = get_time()###开始计时
    start_memory2 = show_info()
    print("开始内存：%fMB" % (start_memory2))

    data_name = 'muraro'
    data_path = 'E:\\Study\\scDCCA-main\\example data\\%s_2000_sample.txt' % data_name
    label_path = 'E:\\Study\\scDCCA-main\\example data\\%s_truelabels1_simple.csv' % data_name

    pretrain_path = 'model_sample/%s_pretrain_param.pth' % data_name
    model_path = 'model_sample/%s_param.pth' % data_name

    # 读取数据
    try:
        x = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)
        y = pd.read_csv(label_path, header=None)[0]
        
        logging.info(f"Data loaded. X shape: {x.shape}, Y shape: {y.shape}")
        
        # 处理标签
        lab = y.unique().tolist()
        logging.info(f"Unique labels: {lab}")
        
        ind = list(range(0, len(lab)))
        mapping = {j: i for i, j in zip(ind, lab)}
        y = y.map(mapping).to_numpy()
        
        # 创建 AnnData 对象
        import scanpy as sc
        adata = sc.AnnData(x)
        adata.obs['Group'] = y

        # 数据预处理
        adata = read_dataset(adata, transpose=False, test_split=False, copy=False)
        adata = normalize(adata, filter_min_counts=False, size_factors=True, 
                          normalize_input=False, logtrans_input=False)

        # 模型参数
        input_size = adata.n_vars
        n_clusters = adata.obs['Group'].unique().shape[0]
        logging.info(f"Input size: {input_size}, Number of clusters: {n_clusters}")

        # 多次运行以获取稳定结果
        cycle = 10
        arii = np.array([])
        nmii = np.array([])
        f11 = np.array([])
        accc = np.array([])
        homoo = np.array([])
        compp = np.array([])

        for i in range(cycle):
            logging.info(f"Running cycle {i+1}/{cycle}")
            
            model = MyModel(
                input_dim=input_size,
                z_dim=32,
                n_clusters=n_clusters,
                encodeLayer=[256, 64],
                decodeLayer=[64,256],
                activation='relu',
                sigma=2.5,
                alpha=1.0,
                gamma=1.0,
                device='cuda:0'
            )

            # 预训练自编码器
            model.pretrain_autoencoder(
                x=adata.X,
                X_raw=adata.raw.X,
                size_factor=adata.obs.size_factors,
                batch_size=1024,
                epochs=70,
                ae_weights=pretrain_path
            )

            # 训练模型
            total_losses, cluster_losses, recon_losses, ssl_losses = model.fit(
                X=adata.X,
                X_raw=adata.raw.X,
                sf=adata.obs.size_factors,
                y=y,
                lr=1.0,
                batch_size=1024,
                num_epochs=100,
                save_path=model_path
            )

            # 计算性能指标
            final_acc, final_f1, final_nmi, final_ari, final_homo, final_comp = evaluate(y, model.y_pred)

            # 记录每次循环的性能指标
            accc = np.append(accc, final_acc)
            f11 = np.append(f11, final_f1)
            arii = np.append(arii, final_ari)
            nmii = np.append(nmii, final_nmi)
            homoo = np.append(homoo, final_homo)
            compp = np.append(compp, final_comp)

        # 输出性能指标
        print('optimal:ACC= {:.4f}'.format(np.max(accc)),', F1= {:.4f}'.format(np.max(f11)), 
              ', NMI= {:.4f}'.format(np.max(nmii)), ', ARI= {:.4f}'.format(np.max(arii)),
              ', HOMO= {:.4f}'.format(np.max(homoo)), ', COMP= {:.4f}'.format(np.max(compp)))
        
        print('mean:ACC= {:.4f}'.format(np.mean(accc)), ', F1= {:.4f}'.format(np.mean(f11)),
              ', NMI= {:.4f}'.format(np.mean(nmii)), ', ARI= {:.4f}'.format(np.mean(arii)),
              ', HOMO= {:.4f}'.format(np.mean(homoo)),', COMP= {:.4f}'.format(np.mean(compp)))
        
        print('std:ACC= {:.4f}'.format(np.std(accc)), ', F1= {:.4f}'.format(np.std(f11)),
              ', NMI= {:.4f}'.format(np.std(nmii)), ', ARI= {:.4f}'.format(np.std(arii)),
              ', HOMO= {:.4f}'.format(np.std(homoo)),', COMP= {:.4f}'.format(np.std(compp)))
        
        print("mymodel", data_name)

        time = get_time() - time_start
        print("Running Time:" + str(time))
        
        zuihou = show_info()
        print("使用内存：%fMB" % (zuihou - start_memory2))
        print("结束内存：%fMB" %(zuihou))
        print("完成")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"Error: {e}")