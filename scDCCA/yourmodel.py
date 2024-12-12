import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from preprocess import *
from collections import defaultdict
from sklearn import preprocessing
from time import time as get_time

import os 
import psutil

'''
这个函数是为了计算消耗内存
'''
def show_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024  / 1024
    return memory

'''
添加构建层的函数
'''
def buildNetwork(layers , activation = "relu"):
    net = []
    for i in range(1 , len(layers)):
        net.append(nn.Linear(layers[i-1],  layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        else:
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class YourModel(nn.Module):
    def __init__(self ,
    input_dim , # 输入维度
    z_dim , # 中间维度
    n_clusters,
    encodeLayer = [] ,
    decodeLayer = [] ,
    activation = "relu" ,
    sigma = 1.0 ,
    alpha = 1.0 ,
    gamma = 1.0 , # 正则化参数
    i_temp = 0.5 , # 对比学习温度
    c_temp = 1.0 , # 聚类学习温度
    i_reg = 0.5 , # 对比学习系数
    c_reg = 0.2 , # 聚类学习系数
    feature_dim = 32 , # 特征维度
    device = 'cpu'
    ):
    super(YourModel , self).__init__()
    self.z_dim = z_dim
    self.n_clusters = n_clusters
    self.activation = activation
    self.sigma = sigma
    self.alpha = alpha
    self.gamma = gamma
    self.encoder = buildNetwork([input_dim]+encodeLayer, activation=activation)###2000-256-64
    self.decoder = buildNetwork([z_dim]+decodeLayer, activation=activation)###32-64-256
    self.decoder1 = buildNetwork([z_dim]+decodeLayer, activation=activation)###32-64-256
    # 均值层
    '''编码器输出 (64维) -> 均值层 -> 隐藏空间 (32维)'''
    self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)###64-32

    self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())##256-2000
    '''.idea\# 从 256 维重建到 2000 维
    # 作用：
    # - 预测基因表达的均值
    # - 使用 MeanAct() 确保均值非负
'''    
    self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())##256-2000
    '''
    计算离散层的函数
    '''
    self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())##256-2000

    self.i_temp = i_temp
    self.c_temp = c_temp
    self.i_reg = i_reg####0.5
    self.c_reg = c_reg####0.2

    self.instance_projector = nn.Sequential(
        nn.Linear(z_dim , z_dim) , 
        nn.ReLU() , 
        nn.Linear(z_dim , feature_dim)
    )

    self.cluster_projector = nn.Sequential(
        nn.Linear(z_dim , n_clusters) , 
        nn.Softmax(dim=1)
    )

    self.mu = nn.Parameter(torch.Tensor(n_clusters , z_dim))
    self.zinb_loss = ZINBLoss()
    self.to(device)


'''
每一行代表一个数据点(cell)
每一列代表属于特定聚类中心的概率
概率越高，表示该数据点越可能属于对应的聚类中心
'''
    def soft_assgin(self , z):
      q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2 , dim=2) / self.alpha)
      q =q**((self.alpha+1.0)/2.0)
      q = (q.t() / torch.sum(q , dim=1)).t()
      return q

   def target_distribution(self , q):
      p   = q**2/q.sum(0)
      return (p.t() / p.sum(1)).t()

   def x_drop(self , x ,p = 0.2):
    # shape[0]样本数(行) shape[1]特征数 
    mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
    mask = torch.vstack(mask_list)
    new_x = x.clone()
    new_x[mask] = 0.0
    return new_x

   def forward(self , x):
    # 添加随机噪声
    z = self._enc_mu(h) ## 64-32
    h = self.decoder(z) ## 32-64-256
    h1 = self.decoder1(z)
    h = (h1 + h) / 2
    
    _mean = self._dec_mean(h)
    _disp = self._dec_disp(h)
    _pi = self._dec_pi(h)
    
    h0 = self.encoder(x)
    z0 = self._enc_mu(h0)
    q = self.soft_assign(z0)
    return z0 , q , _mean , _disp , _pi

   def calc_ssl_lossv1(self , x1 , x2):
    z1 , _ , _ , _ , _ = self.forward(x1)
    z2 , _ , _ , _ , _ = self.forward(x2)
    instance_loss = InstanceLoss(x1.shape[0] , self.i_temp)
    return instance_loss.forward(z1 , z2)

   def calc_ssl_lossv2(self , x1 , x2):
    z1 , q1 , _ , _ , _ = self.forward(x1)
    z2 , q2 , _ , _ , _ = self.forward(x2)
    cluster_loss = ClusterLoss(self.n_clusters , self.c_temp)
    return cluster_loss.forward(q1 , q2)

    def encodeBatch(self , X , batch_size = 256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            encoded = []
            num = X.shape[0]
            num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                inputs = Variable(xbatch)
                z , _ , _ , _ , _ = self.forward(inputs)
                encoded.append(z.data)
            encoded = torch.cat(encoded , dim=0)
            return encoded
   

   '''
   这个损失函数常用于聚类任务中，
   特别是在深度学习的自编码器或者聚类网络中，
   用于衡量两个概率分布之间的差异。KL散度是一种不对称的度量,
   用于衡量一个概率分布相对于另一个概率分布的信息损失。
   '''
   def cluster_loss(self , p  , q) :
    def kld(target , pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)) , dim=-1))
    kldloss = kld(p , q)
    return self.gamma*kldloss

   def pretrain_autoencoder(self , 
    x ,
    X_raw ,
    size_factor ,
    batch_size=256 ,
    lr=0.001 , 
    epochs=400 , 
    ae_save=True , ae_weights='AE_weights.pth.tar'):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        self.cuda()
    dataset = TensorDataset(torch.Tensor(x) , torch.Tensor(X_raw) , torch.Tensor(size_factor))
    dataloader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
    print("Pretraining stage")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad , self.parameters()) , lr=lr)
    for epoch in range(epochs):
        for batch_idx , (x_batch , x_raw_batch , sf_batch) in enumerate(dataloader):
            x_tensor = Variable(x_batch).cuda()
            x_raw_tensor = Variable(x_raw_batch).cuda()
            sf_tensor = Variable(sf_batch).cuda()
            _ , _ , mean_tensor , disp_tensor , pi_tensor = self.forward(x_tensor)
            zinb_loss = self.zinb_loss(x=x_raw_tensor , mean=mean_tensor , disp=disp_tensor , pi=pi_tensor , scale_factor=sf_tensor)
            optimizer.zero_grad()
            zinb_loss.backward()
            optimizer.step()
            print('Pretrain Epoch: [{}] [{}/{} ({:.4f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), zinb_loss.item()))

    if ae_save:
        torch.save(self.state_dict() , as_weights)


def fit(self , X , X_raw , sf , y=None , lr=1. , batch_size=256 , num_epochs=10 , save_path=''):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        self.cuda()
    print('Clustering stage')
    X = torch.tensor(X).cuda()
    X_raw = torch.tensor(X_raw).cuda()
    sf = torch.tensor(sf).cuda()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad , self.parameters()) , lr=lr , rho=.95)

    print('Ininitializing cluster centers with kmeans.')
    kmeans = KMeans(self.n_clusters , n_init=20)
    data = self.encodeBatch(X)
    self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    if y is not None:
        acc , f1 , nmi , ari , homo , comp = evaluate(y , self.y_pred)
        print('Initializing k-means: ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (acc , f1 , nmi , ari , homo , comp))

    self.train()
    num = X.shape[0]
    num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
    acc , nmi , ari , homo , comp , epoch = 0 , 0 , 0 , 0 , 0 , 0
    lst = []
    pred = []
    best_ari = 0.0
    for epoch in range(num_epochs):
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)
        target = self.target_distribution(q)

        self.y_pred = torch.argmax(q , dim = 1).data.cpu().numpy()
        acc , f1 , nmi , ari , homo , comp = evaluate(y , self.y_pred)

        lab_ypred = np.unique(self.y_pred)
        print(lab_ypred)
        
        print('Cluster %d Epoch: [{}] ACC= %.4f, F1= %.4f, NMI= %.4f, ARI= %.4f, HOMO= %.4f, COMP= %.4f' % (epoch , epoch , acc , f1 , nmi , ari , homo , comp))
        pred.append(self.y_pred)
        zhibiao = (acc , f1 , nmi , ari , homo , comp)
        lst.append(zhibiao)
        if ari > best_ari:
            best_ari = ari
            torch.save({'latent': latent, 'q': q, 'p': p}, save_path)
            print('save_successful')


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
data_path = 'E:\Study\scDCCA-main\example data/%s_2000.txt'% data_name
label_path = 'E:\Study\scDCCA-main\example data/%s_truelabel.csv' % data_name
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