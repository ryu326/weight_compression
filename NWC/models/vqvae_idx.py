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

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class IdxEmbedding(nn.Module):
    def __init__(self, num_layers=32, hidden_dim=128, out_dim = 64):
        super().__init__()
        
        self.out_dim = out_dim
        self.embedding = nn.Embedding(num_layers, hidden_dim)  # layer_idx 임베딩
        self.ltype_embedding = nn.Embedding(2, hidden_dim)    # ltype 임베딩
        self.wtype_embedding = nn.Embedding(7, hidden_dim)    # wtype 임베딩
        self.fc = nn.Linear(hidden_dim * 3 + 3, hidden_dim)   # 임베딩 + 숫자 입력
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)          # 예시: 스칼라 출력
        
    def forward(self, x):
        layer_idx_emb = self.embedding(x[:, 0])
        ltype_emb = self.ltype_embedding(x[:, 1])
        wtype_emb = self.wtype_embedding(x[:, 2])
        
        row_col_values = x[:, 3:].float()  # 숫자 값들 (row_idx, col_slice.start, col_slice.stop)
        
        # 결합 및 처리
        combined = torch.cat([layer_idx_emb, ltype_emb, wtype_emb, row_col_values], dim=-1)
        hidden = self.fc(combined)
        output = self.output_layer(hidden)
        return output

class VQVAE_IDX(nn.Module):

    def __init__(
        self, input_size, dim_encoder, n_resblock, n_embeddings, P, dim_embeddings, beta, scale, shift
    ):  # u_length 는 input_dim의 약수로 설정할 것 (안 그러면 에러가 터짐)
        super().__init__()

        self.input_size = input_size
        self.dim_encoder = dim_encoder

        self.dim_encoder_out = dim_embeddings * P
        self.dim_embeddings = dim_embeddings
        self.n_embeddings = n_embeddings

        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)        

        self.encoder = nn.Sequential(
            nn.Linear(input_size, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_encoder_out, dim_encoder),
            ResidualStack(dim_encoder, n_resblock),
            nn.Linear(dim_encoder, input_size),
        )

        self.idx_embedding = IdxEmbedding()
        self.vector_quantization = VectorQuantizer(self.n_embeddings, self.dim_embeddings, beta)

        self.idx_cat = nn.Linear(dim_encoder + self.idx_embedding.out_dim, self.dim_encoder_out)

    def forward(self, x, block_idx_tensor, verbose=False):
        x_shift = (x - self.shift) / self.scale

        out_enc = self.encoder(x_shift)
        out_idx = self.idx_embedding(block_idx_tensor)
        
        z_e = torch.cat([out_enc, out_idx], dim=-1)
        z_e = self.idx_cat(z_e)
        
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
