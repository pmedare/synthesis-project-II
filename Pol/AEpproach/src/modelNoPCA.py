import torch.nn as nn

# Create autoencoder for input vectors of size 65

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(166, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 86),
            nn.LeakyReLU(),
            nn.Linear(86, 48),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(48, 86),
            nn.LeakyReLU(),
            nn.Linear(86, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 166),
            nn.LeakyReLU() 
        )
    
    def forward(self, in_features):
        encoded_features = self.encoder(in_features)
        out = self.decoder(encoded_features)
        return out