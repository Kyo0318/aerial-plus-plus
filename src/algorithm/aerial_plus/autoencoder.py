import torch
import os
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    This variational autoencoder is used to create a numerical representation for the categorical values.
    """

    def __init__(self, data_size, latent_dim=None):
        """
        :param data_size: size of the categorical features in the knowledge graph, after one-hot encoding
        :param latent_dim: dimension of the latent space (default: data_size / 2)
        """
        super().__init__()
        self.data_size = data_size
        self.latent_dim = latent_dim if latent_dim is not None else int(1 * self.data_size / 2)
        
        # Encoder: outputs mean and log variance
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 2)),
            nn.Linear(int(1 * self.data_size / 2), self.latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 2)),
            nn.Linear(int(1 * self.data_size / 2), self.latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(1 * self.data_size / 2)),
            nn.Linear(int(1 * self.data_size / 2), self.data_size)
        )

        self.encoder_mean.apply(self.init_weights)
        self.encoder_logvar.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def save(self, p):
        torch.save(self.encoder_mean.state_dict(), p + 'cat_encoder_mean.pt')
        torch.save(self.encoder_logvar.state_dict(), p + 'cat_encoder_logvar.pt')
        torch.save(self.decoder.state_dict(), p + 'cat_decoder.pt')

    def load(self, p):
        if (os.path.isfile(p + 'cat_encoder_mean.pt') and 
            os.path.isfile(p + 'cat_encoder_logvar.pt') and 
            os.path.isfile(p + 'cat_decoder.pt')):
            self.encoder_mean.load_state_dict(torch.load(p + 'cat_encoder_mean.pt'))
            self.encoder_logvar.load_state_dict(torch.load(p + 'cat_encoder_logvar.pt'))
            self.decoder.load_state_dict(torch.load(p + 'cat_decoder.pt'))
            self.encoder_mean.eval()
            self.encoder_logvar.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices):
        # Encode to latent distribution parameters
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mean, logvar)
        
        # Decode
        y = self.decoder(z)

        # Split the tensor into chunks based on the ranges
        chunks = [y[:, start:end] for start, end in input_vector_category_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        # Return reconstructed output, mean, and logvar for VAE training
        return y, mean, logvar
    
    def compute_kl_loss(self, mean, logvar):
        """
        Compute KL divergence loss: KL(N(mean, var) || N(0, 1))
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()
