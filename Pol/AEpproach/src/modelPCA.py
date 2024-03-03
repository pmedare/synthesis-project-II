import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(65, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 65),
            nn.LeakyReLU() 
        )
    
    def forward(self, in_features):
        encoded_features = self.encoder(in_features)
        out = self.decoder(encoded_features)
        return out