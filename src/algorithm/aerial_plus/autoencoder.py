import torch
import os
import math
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """
    Transformer-based Autoencoder for tabular data.
    Each feature (categorical variable) is treated as a token.
    """

    def __init__(self, input_vector_category_indices, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        """
        :param input_vector_category_indices: List of tuples [(start, end), ...] indicating indices for each feature.
                                              Necessary to build per-feature projection layers.
        :param d_model: The dimension of the transformer embeddings (latent space per feature).
        :param nhead: Number of heads in the multiheadattention models.
        :param num_layers: Number of sub-encoder-layers in the transformer encoder.
        """
        super().__init__()
        
        self.input_vector_category_indices = input_vector_category_indices
        self.num_features = len(input_vector_category_indices)
        self.d_model = d_model

        # Calculate the size of each category (e.g., feature 1 has 3 categories, feature 2 has 5...)
        # This is needed to project each one-hot segment to d_model size.
        self.category_counts = [end - start for start, end in input_vector_category_indices]
        
        # --- Encoder Part ---
        # 1. Feature Tokenizers: Project each one-hot feature to d_model dimension
        self.feature_tokenizers = nn.ModuleList([
            nn.Linear(c_size, d_model) for c_size in self.category_counts
        ])

        # 2. Positional Encoding: Learnable embedding to distinguish between Feature A and Feature B
        # Since tabular data doesn't have sequential order, we use a learnable embedding per feature index.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_features, d_model))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Expected input: (Batch, Seq_len, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder Part ---
        # Project back from d_model to original one-hot size for reconstruction
        self.feature_reconstructors = nn.ModuleList([
            nn.Linear(d_model, c_size) for c_size in self.category_counts
        ])

        # Weight Initialization
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        Xavier initialization
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def save(self, p):
        # Save the entire state dict which includes all sub-modules
        torch.save(self.state_dict(), p + 'transformer_ae.pt')

    def load(self, p):
        path = p + 'transformer_ae.pt'
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices=None):
        """
        :param x: Flat one-hot encoded vector (Batch, Total_Categories)
        :param input_vector_category_indices: (Optional in forward if provided in init) 
               Used here to slice the input x.
        """
        # Ideally use the indices stored at init, but allow override if passed
        indices = input_vector_category_indices if input_vector_category_indices else self.input_vector_category_indices
        
        batch_size = x.size(0)

        # --- 1. Tokenization (Embedding) ---
        # Slicing the flat vector x and projecting each feature to d_model
        # Resulting shape: List of (Batch, d_model) -> Stack to (Batch, Num_Features, d_model)
        tokens = []
        for i, (start, end) in enumerate(indices):
            feature_slice = x[:, start:end]
            token = self.feature_tokenizers[i](feature_slice)
            tokens.append(token)
        
        x_emb = torch.stack(tokens, dim=1) # (Batch, Num_Features, d_model)

        # --- 2. Add Positional Embeddings ---
        x_emb = x_emb + self.pos_embedding

        # --- 3. Transformer Encoder ---
        # Latent representation
        z = self.transformer_encoder(x_emb) # (Batch, Num_Features, d_model)

        # --- 4. Reconstruction ---
        # Project back to original dimensions and apply Softmax per feature
        outputs = []
        for i, (start, end) in enumerate(indices):
            # z[:, i, :] is the latent vector for feature i
            recon = self.feature_reconstructors[i](z[:, i, :]) # (Batch, Category_Count_i)
            outputs.append(F.softmax(recon, dim=1))

        # --- 5. Concatenate ---
        # Back to flat vector (Batch, Total_Categories)
        y = torch.cat(outputs, dim=1)

        return y