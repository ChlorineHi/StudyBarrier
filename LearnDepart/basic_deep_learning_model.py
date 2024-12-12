import torch
import torch.nn as nn
import torch.optim as optim
#  mymodel  model
#  input_dim  input  z_dim  z   n_clusters   

class MyModel(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        self.cluster_projector = nn.Linear(z_dim, n_clusters)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return z, reconstructed

    def fit(self, data_loader, num_epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for batch in data_loader:
                inputs = batch[0] # 
                optimizer.zero_grad()
                z, reconstructed = self.forward(inputs)
                loss = self.compute_loss(inputs, reconstructed)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def compute_loss(self, inputs, reconstructed):
        return nn.MSELoss()(reconstructed, inputs)
