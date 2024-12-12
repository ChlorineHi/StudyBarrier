import torch
import torch.nn as nn
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        # 对角线设置为0
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        # 将两组嵌入向量（z_i和z_j）沿第0维（批次维度）拼接
        # 例如：如果z_i和z_j都是[256, 128]，拼接后变成[512, 128]
        z = torch.cat((z_i, z_j), dim=0)
        # 计算嵌入向量的相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature
        # 提取相似度
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # 将相似度矩阵转换为正样本和负样本
        # 收集所有正样本对的相似度
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # 收集所有负样本对的相似度
        # 自动的计算所用的mask
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):###定义ClusterLoss这个类的基本参数和方法
        super(ClusterLoss, self).__init__()
        self.class_num = class_num###类别数目，比如deng是10
        self.temperature = temperature###温度参数=1.0
        self.mask = self.mask_correlated_clusters(class_num)###定义mask的方式
        self.criterion = nn.CrossEntropyLoss(reduction="sum")###定义损失标准采用交叉熵
        self.similarity_f = nn.CosineSimilarity(dim=2)###定义相似性采用余弦相似度

    def mask_correlated_clusters(self, class_num):
        # 生成一个掩膜
        N = 2 * class_num
        mask = torch.ones((N, N))###生成(N, N)的全1矩阵
        mask = mask.fill_diagonal_(0)###对角线元素置0
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()###转成bool型变量，即其中的1变成True
        return mask

    def forward(self, c_i, c_j):##对每一批，c_i是256个10维的向量，计算了每个cluster的熵
        # p_i是每个cluster的概率向量，是一个10维的向量
        p_i = c_i.sum(0).view(-1)###把这一批256个求和，得到一个总的p_i
        p_i /= p_i.sum()###p_i.sum()=256,所以p_i这里是求平均
        # ne_i是每个cluster的熵
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()###log(p_i)求得是以e为底的ln(p_i)
        # p_j是每个cluster的概率向量，是一个10维的向量
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        # ne_j是每个cluster的熵
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        # ne_loss是所有cluster的熵的和
        ne_loss = ne_i + ne_j

        # c_i和c_j是10*256的矩阵
        c_i = c_i.t()####转置
        c_j = c_j.t()###转置
        # N是总的cluster数目
        N = 2 * self.class_num###N=20
        # c是20*256的矩阵，是c_i和c_j的拼接
        c = torch.cat((c_i, c_j), dim=0)##拼接

        # sim是20*20的矩阵，是c和c的相似度
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature###20*20的
        # sim_i_j是对角线元素，是positive sample的相似度
        sim_i_j = torch.diag(sim, self.class_num)###取矩阵的对角线元素
        # sim_j_i是对角线元素，是positive sample的相似度
        sim_j_i = torch.diag(sim, -self.class_num)

        # positive_clusters是20*1的矩阵，是positive sample的相似度
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_clusters是20*19的矩阵，是negative sample的相似度
        negative_clusters = sim[self.mask].reshape(N, -1)

        # labels是20个0的向量
        labels = torch.zeros(N).to(positive_clusters.device).long()
        # logits是20*20的矩阵，是positive sample和negative sample的相似度
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        # loss是对比损失
        loss = self.criterion(logits, labels)
        # loss /= N
        # loss是对比损失加上熵损失
        return loss + ne_loss