import math

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x


# U를 만들어줄 뉴럴넷 후보들. 아이디어 잘 짜서 추가를 해보도록 하자.
import torch.nn as nn


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
        self.stack = nn.ModuleList([Linear_ResBlock(in_dim)] * n_res_layers)

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

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        ) # 중요한 애들에 대해서 값을 높이자. 그러면 전체 로스에서 '그 애들을 따라잡는 로스'의 비중이 높아지니까, 코드북 역시 '중요한 애들' 위주로 형성되지 않을까? (될거 같다)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        
        # ########################################################
        # # 아이디어
        # loss_per_ele_a = (z_q.detach() - z) ** 2
        # loss_per_ele_b = (z_q - z.detach()) ** 2
        
        # # scale = (AWQ 에서 얻은 정보) <- 중요한 애들은 1.0에 가깝고, 덜 중요한 애들은 0.0에 가까울거 같다. 
        # # def get_act_scale(x):
        # #     return x.abs().view(-1, x.shape[-1]).mean(0)
        
        # loss_per_ele_a = loss_per_ele_a * scale
        # loss_per_ele_b = loss_per_ele_b * scale
        
        # loss = torch.mean(loss_per_ele_a + self.beta * torch.mean(loss_per_ele_b))
        # ########################################################
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_dim, n_res_layers, dim_encoder, dim_encoder_out):
        super(Encoder, self).__init__()
        
        
        self.weight_in = nn.Linear(in_dim, dim_encoder)
        self.act_in = nn.Linear(in_dim, dim_encoder)
        self.weight_stack = nn.ModuleList([Linear_ResBlock(dim_encoder)] * n_res_layers)
        self.act_stack = nn.ModuleList([Linear_ResBlock(dim_encoder)] * n_res_layers)
        self.out = nn.Linear(dim_encoder, dim_encoder_out)

    def forward(self, x, a):
        a = self.act_in(a)
        x = self.weight_in(x)
        
        a_outs = []        
        for layer in self.act_stack:
            a = layer(a)
            a_outs.append(a)
        
        for i, layer in enumerate(self.weight_stack):
            x = layer(x)
            x = x * a_outs[i]
        return self.out(x)

class VQVAE_MAG(nn.Module):
    def __init__(
        self, input_size, dim_encoder, n_resblock, n_embeddings, P, dim_embeddings, beta, scale, shift):
        super().__init__()

        self.input_size = input_size
        self.dim_encoder = dim_encoder

        self.dim_encoder_out = dim_embeddings * P
        self.dim_embeddings = dim_embeddings
        self.n_embeddings = n_embeddings

        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)        

        self.encoder = Encoder(input_size, n_resblock, self.dim_encoder, self.dim_encoder_out)

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_encoder_out, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, input_size),
        )

        self.vector_quantization = VectorQuantizer(self.n_embeddings, self.dim_embeddings, beta)

    def forward(self, data, verbose=False):
        # {'block_idx': block_idx, 
        # 'tensor_block_idx': t_block_idx,
        # 'weight_block': weight_block,
        # 'input_block': input_block }
                
        x = data['weight_block']
        ch_mag = data['input_block']
        
        x_shift = (x - self.shift) / self.scale

        z_e = self.encoder(x_shift, ch_mag)

        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e)

        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            assert False

        x_hat = self.scale * x_hat + self.shift

        return {
            "embedding_loss": embedding_loss,
            "x": x,
            "x_hat": x_hat,
            "perplexity": perplexity,
            "z_q": z_q,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
        }
