import torch
import torch.nn as nn
from layers import MeanAct, DispAct

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
        self.encoder = self.buildNetwork([input_dim]+encodeLayer, activation=activation)
        self.decoder = self.buildNetwork([z_dim]+decodeLayer, activation=activation)
        self.decoder1 = self.buildNetwork([z_dim]+decodeLayer, activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.i_temp = i_temp
        self.c_temp = c_temp
        self.i_reg = i_reg
        self.c_reg = c_reg
        self.instance_projector = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, feature_dim))
        self.cluster_projector = nn.Sequential(
            nn.Linear(z_dim, n_clusters),
            nn.Softmax(dim=1))
        self.mu = nn.Parameter(torch.Tensor(n_clusters, z_dim))
        self.to(device)

    def buildNetwork(self, layers, activation="relu"):
        net = []
        for i in range(1, len(layers)):
            net.append(nn.Linear(layers[i-1], layers[i]))
            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
        return nn.Sequential(*net)

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        h1 = self.decoder1(z)
        h = (h1 + h) / 2
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi
