from sklearn.cluster import KMeans
from contrastive_loss import ClusterLoss, InstanceLoss
class ClusteringModel:
    """
    聚类模型

    Parameters:
    - input_dim: 输入维度
    - z_dim: 中间维度
    - n_clusters: 聚类数量
    - encodeLayer: 编码器网络结构
    - decodeLayer: 解码器网络结构
    """
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[]):
        self.n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(z_dim, n_clusters),
        )
        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))

    def compute_cluster_loss(self, q1, q2):
        """
        计算聚类Loss

        Parameters:
        - q1: 第一个视图的聚类概率
        - q2: 第二个视图的聚类概率

        Returns:
        - cluster_loss: 聚类Loss
        """
        cluster_loss = ClusterLoss(self.n_clusters, self.c_temp)
        return cluster_loss.forward(q1, q2)

    def initialize_kmeans(self):
        """
        使用kmeans对聚类中心初始化
        """
        kmeans = KMeans(self.n_clusters, n_init=20)
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

    def train_cluster_loss(self, target, qbatch):
        """
        训练聚类Loss

        Parameters:
        - target: label
        - qbatch: 聚类概率

        Returns:
        - cluster_loss: 聚类Loss
        """
        cluster_loss = self.cluster_loss(target, qbatch)
        return cluster_loss
