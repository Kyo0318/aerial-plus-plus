import torch
import os
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    This deep autoencoder is used to create a numerical representation for the categorical values.
    It uses multiple hidden layers for both encoder and decoder to capture complex patterns.
    """

    def __init__(self, data_size):
        """
        :param data_size: size of the categorical features in the knowledge graph, after one-hot encoding
        """
        super().__init__()
        self.data_size = data_size
        
        # Calculate layer sizes for deep architecture
        hidden_size_1 = int(self.data_size / 2)
        hidden_size_2 = int(self.data_size / 4)
        hidden_size_3 = int(self.data_size / 8)
        latent_size = max(int(self.data_size / 16), 16)  # Ensure minimum size
        
        # Deep encoder with multiple layers and activation functions
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, hidden_size_1),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_1),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_2),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_3),
            nn.Linear(hidden_size_3, latent_size),
        )
        
        # Deep decoder with multiple layers and activation functions (symmetric to encoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size_3),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_3),
            nn.Linear(hidden_size_3, hidden_size_2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_2),
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(),
            nn.LayerNorm(hidden_size_1),
            nn.Linear(hidden_size_1, self.data_size)
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        LayerNorm layers are initialized with default parameters.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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
        y = self.encoder(x)
        y = self.decoder(y)

        # Split the tensor into chunks based on the ranges
        chunks = [y[:, start:end] for start, end in input_vector_category_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        return y
