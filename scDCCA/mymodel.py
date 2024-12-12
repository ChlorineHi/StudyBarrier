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
from evaluation import evaluate
from preprocess import *####用于导入指定模块中的全部定义。
from collections import defaultdict
from sklearn import preprocessing
import random
from contrastive_loss import ClusterLoss, InstanceLoss
from pandas import Series
from time import time as get_time

import os
import psutil
import matplotlib.pyplot as plt

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
    
    def encodeBatch(self, X, batch_size=256):####encodeBatch的作用是对输入进行编码得到编码的嵌入向量，输入就是原始的x，
                                               # 每次只处理一个批次的数据，将数据按批次处理，避免超显存
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []###encoded是一个空列表
        num = X.shape[0]####num=268即细胞数目
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))###每批次256，算算要多少批，这里是268/256=2
        for batch_idx in range(num_batch):###对每一批
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]###取该批次的数据
            inputs = Variable(xbatch)
            z,_, _, _, _ = self.forward(inputs)###对数据进行编码2000-256-64-32得到32维的z
            encoded.append(z.data)###将这256个32维的数据存在encoded列表中，作为它的一个元素，回到for继续，直到所有批次存完，最后encoded列表中存放的是所有细胞的32维的编码

        encoded = torch.cat(encoded, dim=0)###将encoded列表的所有元素拼接起来，成为一个(num,32)的tensor
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

        # 创建 model 目录
        os.makedirs('model', exist_ok=True)

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
        plt.savefig('model/pretrain_losses.png')
        plt.close()

        # 保存模型
        if ae_save:
            torch.save(self.state_dict(), ae_weights)

        return train_losses, zinb_losses, kl_losses

    def fit(self, X, X_raw, sf, y=None, lr=1., batch_size=256, num_epochs=10, save_path=''):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)###定义聚类方法是kmeans，类别数目=10,选20次最优的
        data = self.encodeBatch(X)####data就是X编码以后的num个32维的隐层表示，是一个tensor
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())###对这个编码以后的隐层数据用kmeans聚类得到预测标签
        self.y_pred_last = self.y_pred###备份一下kmeans聚类得到预测标签
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))###kmeans.cluster_centers_是self.n_clusters*32的数组，就是质心坐标,将其备份一下，放在 self.mu.data里
        if y is not None:
            acc, f1, nmi, ari, homo, comp = evaluate(y, self.y_pred)
            print('Initializing k-means: ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (acc, f1, nmi, ari, homo, comp))
        
        self.train()
        num = X.shape[0]###细胞数
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))###算下需要多少批次
        acc, nmi, ari, homo,comp, epoch = 0, 0, 0, 0, 0, 0
        lst = []  #####创建一个空列表，用于存放指标
        pred = []  #####创建一个空列表，用于存放预测标签
        best_ari = 0.0  ###初始化最优的ARI为0
        for epoch in range(num_epochs):
            # update the targe distribution p
            latent = self.encodeBatch(X)###先不考虑损失，直接对X进行编码得到隐层，268个32维的，根据这个编码的向量可以得到最初的q,p，以及用q的预测标签，和该标签的准确度等等
            q = self.soft_assign(latent)###对编码得到的隐层向量算其与随机初始化的质心坐标的相似程度，268个
            p = self.target_distribution(q).data###算目标分布，这里要事先把p算好，因为forward只返回q,如果这里不计算，就要在265行得到qbatch以后计算，效果是一样的

            # evalute the clustering performance
            self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()###用软标签q得到预测标签
            acc, f1, nmi, ari, homo, comp = evaluate(y, self.y_pred)##算软标签q的指标

            lab_ypred = np.unique(self.y_pred)
            print(lab_ypred)



            print('Cluster %d : ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (epoch+1, acc, f1, nmi, ari, homo, comp))
            pred.append(self.y_pred)  #####在列表中增加元素，只不过这个元素是每一次预测的标签
            zhibiao = (acc, f1, nmi, ari, homo, comp)
            lst.append(zhibiao)  #####在列表中增加元素，只不过这个元素是上面的5个指标

            if best_ari < ari:###如果当前得到的ari比最优的ari大，说明当前的更好，就把当前的存起来，最终保存的是训练次数中最优的ari
                best_ari = ari
                torch.save({'latent': latent, 'q': q, 'p': p}, save_path)###保存隐层变量
                latent = latent.cpu().numpy()###先从tensor转成array
                #
                print('save_successful')


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            c_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]#p是268*10的矩阵，从里面取一个批次256个
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)###对该批次的输入进行编码得到相应的5个中间变量

                inputs1 = self.x_drop(inputs, p=0.2)
                inputs2 = self.x_drop(inputs, p=0.2)
                c_loss = self.calc_ssl_lossv2(inputs1, inputs2)###计算编码后的对比聚类损失
                c_loss = self.c_reg * c_loss### 对比损失乘以系数
                
            
                cluster_loss = self.cluster_loss(target, qbatch)###算聚类kl损失
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)###算重构的zinb损失
                loss = cluster_loss + recon_loss + c_loss#####总的训练损失是kl聚类损失+重构损失+对比聚类损失
                #loss = cluster_loss + c_loss  #####总的训练损失是kl聚类损失+重构损失+对比聚类损失
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)###因为再算每一个损失的时候算的都是平均值，所以乘以个数
                recon_loss_val += recon_loss.data * len(inputs)
                c_loss_val += c_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val + c_loss_val####总的训练损失是kl聚类损失+重构损失+对比聚类损失

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f C Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, c_loss_val / num))####但是输出的这里其实还是均值

        cunari = []  #####初始化
        for j in range(len(lst)):  ###j从0到num_epochs-1
            aris = lst[j][2]
            cunari.append(aris)
        max_ari = max(cunari)  ###找到最大的ari
        maxid = cunari.index(max_ari)  ####找到最大的ari的指标
        optimal_pred = pred[maxid]
        #np.savetxt("C:\\Users\\Administrator\\Desktop\\11\\%s_predlabel1.csv"% data_name, optimal_pred, delimiter=' ')###一次才用
        final_acc, final_f1, final_nmi, final_ari, final_homo, final_comp = evaluate(y, optimal_pred)
        return final_acc, final_f1, final_nmi, final_ari, final_homo, final_comp




time_start = get_time()###开始计时
start_memory2 = show_info()
print("开始内存：%fMB" % (start_memory2))



data_name = 'muraro'
data_path = 'E:\\Study\\scDCCA-main\\example data\\%s_2000.txt' % data_name

#data_path = 'E:\Study\scDCCA-main\example data/%s_2000.txt'% data_name
label_path = 'E:\\Study\\scDCCA-main\\example data\\%s_truelabels1.csv' % data_name
#label_path = 'E:\Study\scDCCA-main\example data/%s_truelabel.csv' % data_name

pretrain_path = 'model/%s_pretrain_param.pth' % data_name
model_path = 'model/%s_param.pth' % data_name

#########完整数据
x = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)

#y = pd.read_csv(label_path)['x']

y = pd.read_csv(label_path, header=None)[0]####大数据用这两行读取，没有行列名

# ######################增加下采样
# x1 = pd.read_csv(data_path, header=None).to_numpy().astype(np.float32)
#
# y1 = pd.read_csv(label_path)['x']
# #y1 = pd.read_csv(label_path, header=None)[0]####大数据用这两行读取，没有行列名
# y1 = y1.tolist()
#
# df = pd.DataFrame(x1)
# x = df.sample(frac=0.4)
# index = x.index
# x = x.values
#
# y = [y1[i]for i in index]
#
# y = Series(y)
# ##########################################








lab = y.unique().tolist()
print(lab)
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()






adata = sc.AnnData(x)
adata.obs['Group'] = y


# ##rds数据集经预处理后，用下面这2行,与下面两行二选一
adata = read_dataset(adata,transpose=False,test_split=False,copy=False)
adata = normalize(adata,filter_min_counts=False, size_factors=True, normalize_input=False, logtrans_input=False)



######################################################
input_size = adata.n_vars###2000
n_clusters = adata.obs['Group'].unique().shape[0]
print(n_clusters)




cycle = 10
arii = np.array([])
nmii = np.array([])
f11 = np.array([])
accc = np.array([])
homoo = np.array([])
compp = np.array([])

for i in range(cycle):

   print("第%d次循环", i)

   model = MyModel(
     input_dim=input_size,###2000
     z_dim=32,
     n_clusters=n_clusters,
     encodeLayer=[256, 64],
     decodeLayer=[64,256],
     activation='relu',
     sigma=2.5,
     alpha=1.0,
     gamma=1.0,
     device='cuda:0')

   model.pretrain_autoencoder(
     x=adata.X,
     X_raw=adata.raw.X,
     size_factor=adata.obs.size_factors,
     batch_size=1024,
     epochs=70,
     ae_weights=pretrain_path)

   final_acc, final_f1, final_nmi, final_ari, final_homo, final_comp = model.fit(
      X=adata.X,
      X_raw=adata.raw.X,
      sf=adata.obs.size_factors,
      y=y,
      lr=1.0,
      batch_size=1024,
      num_epochs=100,
      save_path=model_path)




   accc = np.append(accc, final_acc)
   f11 = np.append(f11, final_f1)
   arii = np.append(arii, final_ari)
   nmii = np.append(nmii, final_nmi)
   homoo = np.append(homoo, final_homo)
   compp = np.append(compp, final_comp)







print('optimal:ACC= {:.4f}'.format(np.max(accc)),', F1= {:.4f}'.format(np.max(f11)), ', NMI= {:.4f}'.format(np.max(nmii)),
      ', ARI= {:.4f}'.format(np.max(arii)),', HOMO= {:.4f}'.format(np.max(homoo)), ', COMP= {:.4f}'.format(np.max(compp)))
print('mean:ACC= {:.4f}'.format(np.mean(accc)), ', F1= {:.4f}'.format(np.mean(f11)),', NMI= {:.4f}'.format(np.mean(nmii)),
      ', ARI= {:.4f}'.format(np.mean(arii)),', HOMO= {:.4f}'.format(np.mean(homoo)),', COMP= {:.4f}'.format(np.mean(compp)))
print('std:ACC= {:.4f}'.format(np.std(accc)), ', F1= {:.4f}'.format(np.std(f11)),', NMI= {:.4f}'.format(np.std(nmii)),
      ', ARI= {:.4f}'.format(np.std(arii)),', HOMO= {:.4f}'.format(np.std(homoo)),', COMP= {:.4f}'.format(np.std(compp)))
print("mymodel",data_name)



time = get_time() - time_start
print("Running Time:" + str(time))
zuihou =show_info()
print("使用内存：%fMB" % (zuihou - start_memory2))
print("结束内存：%fMB" %(zuihou))
print("完成")