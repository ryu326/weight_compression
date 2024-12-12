## V2 : coefficient 를 nn으로 만드는게 아니라 U inverse * X 로 만들기

from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel

import torch, math
import torch.nn as nn 

def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x

class Linear_ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        
        self.lin_1 = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.LayerNorm(in_ch),
            nn.ReLU(),
        )
    
    def forward(self, x):
        identity = x
        res = self.lin_1(x)      
        out = identity + res
        
        return out
    
class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [Linear_ResBlock(in_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)
        z.shape = (batch, P, channel)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class VQ_SEEDLM_v2(nn.Module):
        
    def __init__(self, input_size, dim_encoder, n_resblock, n_embeddings, P, beta, scale, shift): # u_length 는 input_dim의 약수로 설정할 것 (안 그러면 에러가 터짐)
        super().__init__()
        
        self.input_size = input_size
        self.dim_encoder = dim_encoder
        self.P = P
        
        # self.dim_encoder_out = input_size * P + P ## P개의 embedding, 1개의 coefficients
        self.dim_encoder_out = input_size * P ## P개의 embedding
        self.dim_embeddings = input_size
        self.n_embeddings = n_embeddings

        self.scale = scale  # dataset_std
        self.shift = shift  # dataset_mean
        
        self.encoder = nn.Sequential(
                nn.Linear(input_size, dim_encoder),
                ResidualStack(dim_encoder, n_resblock),
                nn.Linear(dim_encoder, self.dim_encoder_out), 
            )
        
        self.vector_quantization = VectorQuantizer(
                self.n_embeddings, self.dim_embeddings, beta
            )
    
    def forward(self, x, verbose=False):
        x_shift = (x - self.shift) / self.scale
        
        z = self.encoder(x_shift)
        z_e = z.contiguous().view(-1, self.P, self.dim_embeddings)
        
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(
            z_e)
        
        # import ipdb; ipdb.set_trace()
        coefficient = torch.matmul(torch.linalg.pinv(z_q.permute(0, 2, 1).contiguous()), x_shift.unsqueeze(-1))
        
        ## (b, dim_embedding, P) @ (b, P, 1) --> (b, dim_embedding, 1)
        x_hat  = torch.matmul(z_q.permute(0, 2, 1).contiguous(), coefficient)
        x_hat = x_hat.squeeze(-1)        

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        x_hat = self.scale * x_hat + self.shift

        return {'embedding_loss': embedding_loss,
                'x': x,
                'x_hat': x_hat, 
                'perplexity': perplexity,
                'z_q':z_q,
                'min_encodings':min_encodings,
                'min_encoding_indices':min_encoding_indices,
                'coefficient':coefficient}