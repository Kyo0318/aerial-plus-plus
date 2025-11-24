import torch
import os
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    Sparse Autoencoder with KL divergence regularization for sparsity constraint.
    This autoencoder is used to create a numerical representation for the categorical values.
    """

    def __init__(self, data_size, sparsity_param=0.05, beta=0.1):
        """
        :param data_size: size of the categorical features in the knowledge graph, after one-hot encoding
        :param sparsity_param: target average activation of hidden units (rho), typically 0.05
        :param beta: weight of the sparsity penalty term
        """
        super().__init__()
        self.data_size = data_size
        self.sparsity_param = sparsity_param  # ρ (rho)
        self.beta = beta  # β - weight of sparsity penalty
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 2)),
            nn.Sigmoid()  # Add activation for sparsity constraint
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(1 * self.data_size / 2), self.data_size)
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        
        # For tracking hidden layer activations
        self.hidden_activations = None

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, p):
        torch.save(self.encoder.state_dict(), p + 'cat_encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'cat_decoder.pt')

    def load(self, p):
        if os.path.isfile(p + 'cat_encoder.pt') and os.path.isfile(p + 'cat_decoder.pt'):
            self.encoder.load_state_dict(torch.load(p + 'cat_encoder.pt'))
            self.decoder.load_state_dict(torch.load(p + 'cat_decoder.pt'))
            self.encoder.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices):
        # Encode and store hidden activations for sparsity calculation
        hidden = self.encoder(x)
        self.hidden_activations = hidden  # Store for KL divergence calculation
        
        # Decode
        y = self.decoder(hidden)

        # Split the tensor into chunks based on the ranges
        chunks = [y[:, start:end] for start, end in input_vector_category_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        return y
    
    def kl_divergence_loss(self):
        """
        Calculate KL divergence for sparsity constraint.
        KL(ρ || ρ̂) = ρ * log(ρ/ρ̂) + (1-ρ) * log((1-ρ)/(1-ρ̂))
        where ρ is the target sparsity and ρ̂ is the average activation
        """
        if self.hidden_activations is None:
            return torch.tensor(0.0)
        
        # Calculate average activation across batch for each hidden unit
        rho_hat = torch.mean(self.hidden_activations, dim=0)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        rho = self.sparsity_param
        
        # KL divergence
        kl_div = rho * torch.log((rho + eps) / (rho_hat + eps)) + \
                 (1 - rho) * torch.log((1 - rho + eps) / (1 - rho_hat + eps))
        
        return torch.sum(kl_div)
