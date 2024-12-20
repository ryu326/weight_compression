from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel

import torch, math
import torch.nn as nn


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x


# U를 만들어줄 뉴럴넷 후보들. 아이디어 잘 짜서 추가를 해보도록 하자.
import torch.nn as nn

# MLP mixer 맹키로 짜봄
# 사각형 데이터 처리하고 데이터 간 연관성이란 걸 우린 알지 못하니 연산을 양 쪽(?)에 골고루 맥여야 할거 같았음

# class Linear_ResBlock(nn.Module):
#     def __init__(self, dim_1, dim_2):
#         super().__init__()

#         self.lin_1 = nn.Sequential(
#             nn.Linear(dim_1, dim_1//2),
#             nn.LayerNorm(dim_1//2),
#             nn.ReLU(),
#             nn.Linear(dim_1//2, dim_1)
#         )

#         self.lin_2 = nn.Sequential(
#             nn.Linear(dim_2, dim_2//2),
#             nn.LayerNorm(dim_2//2),
#             nn.ReLU(),
#             nn.Linear(dim_2//2, dim_2)
#         )

#     def forward(self, x):

#         identity = x

#         res = self.lin_1(x) # [b, d1, d2]
#         res = res.permute(0,2,1).contiguous() # [b, d2, d1]
#         res = self.lin_2(res)
#         res = res.permute(0,2,1).contiguous() # [b, d1, d2]

#         out = identity + res

#         return out

# class making_U_MLP(nn.Module):
#     def __init__(self, dim, u_length):
#         super().__init__()

#         self.nn = nn.Sequential(
#             Linear_ResBlock(dim, dim),
#             Linear_ResBlock(dim, dim),
#             Linear_ResBlock(dim, dim),
#             Linear_ResBlock(dim, dim),
#             nn.Linear(dim, u_length), # [dim, dim] -> [dim, u_length]
#             Linear_ResBlock(u_length, dim),
#             Linear_ResBlock(u_length, dim),
#             Linear_ResBlock(u_length, dim),
#             Linear_ResBlock(u_length, dim)
#         )

#     def forward(self, x):
#         out = self.nn(x)

#         return out


class Learnable_SEEDLM(nn.Module):

    def __init__(self, input_dim=512, u_length=4):  # u_length 는 input_dim의 약수로 설정할 것 (안 그러면 에러가 터짐)
        super().__init__()

        self.input_dim = input_dim

        # U 만들어주는 뉴럴넷
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)
        self.network = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # (1, input_dim//4) 영역 중 젤 큰 값을 고름. 값 고르는 틀(?)은 매번 (1, input_dim//4) 만큼 이동함 (CNN 작동 알고리즘 보샘)
        self.pooling = nn.MaxPool2d((1, input_dim // u_length), stride=(1, input_dim // u_length))

        # self.network = making_U_MLP(input_dim, u_length)

    def update(self, scale_table=None, force=False):

        updated = super().update(force=force)
        return updated

    def forward(self, x):

        u = self.network(x)  # U를 제작
        u = self.pooling(u)  # 정사각형에서 직사각형으로, (d, d) -> (d, 4)

        t = torch.matmul(x, u)  # t 제작

        u_inv = torch.linalg.pinv(u)
        x_hat = torch.matmul(t, u_inv)

        return {
            "x": x,
            "x_hat": x_hat,
            # "likelihoods": {"t": t_likelihoods, "u": u_likelihoods}
        }

    # def compress(self, x):

    #     u = self.network(x) # U를 제작
    #     t = torch.matmul(x, u) # t 제작

    #     u = u.unsqueeze(-1)
    #     t = t.unsqueeze(-1)

    #     u_strings = self.entropy_bottleneck_u.compress(u)
    #     t_strings = self.entropy_bottleneck_t.compress(t)

    #     return {"strings": [u_strings, t_strings], "shapes": [u.size(), t.size()]}

    # def decompress(self, strings, shapes):

    #     u_hat = self.entropy_bottleneck_u.decompress(strings[0], shapes[0])
    #     t_hat = self.entropy_bottleneck_t.decompress(strings[1], shapes[1])

    #     u_hat = u_hat.squeeze(-1)
    #     t_hat = t_hat.squeeze(-1)

    #     u_hat_inv = torch.linalg.pinv(u_hat)
    #     x_hat = torch.matmul(t_hat, u_hat_inv)

    #     return {"x_hat": x_hat}
