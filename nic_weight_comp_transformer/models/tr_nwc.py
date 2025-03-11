import math

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)].detach()


class TR_NIC(CompressionModel):

    def __init__(self, input_dim=512, N=256, M=1024, num_slices=4, max_support_slices=5, **kwargs):
        super().__init__()

        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.dim = N
        self.M = M

        # self.positional_encoding = PositionalEncoding(d_model)
        # x = self.positional_encoding(x)

        begin = 0

        g_a_layer = nn.TransformerEncoderLayer(d_model=N, nhead=8, batch_first=True)

        self.g_a = nn.Sequential(
            nn.Linear(input_dim, N),
            nn.TransformerEncoder(g_a_layer, num_layers=8),  # 얘가 받는 데이터 형식 (batch, seq_len, seq_dim)
            nn.Linear(N, M),
        )

        g_s_layer = nn.TransformerEncoderLayer(d_model=N, nhead=8, batch_first=True)
        self.g_s = nn.Sequential(
            nn.Linear(M, N), nn.TransformerEncoder(g_s_layer, num_layers=8), nn.Linear(N, input_dim)
        )

        h_a_layer = nn.TransformerEncoderLayer(d_model=N, nhead=8, batch_first=True)
        self.h_a = nn.Sequential(nn.Linear(M, N), nn.TransformerEncoder(h_a_layer, num_layers=2))

        h_mean_s_layer = nn.TransformerEncoderLayer(d_model=N, nhead=8, batch_first=True)
        self.h_mean_s = nn.Sequential(
            nn.TransformerEncoder(h_mean_s_layer, num_layers=2),
            nn.Linear(N, M),
        )

        h_scale_s_layer = nn.TransformerEncoderLayer(d_model=N, nhead=8, batch_first=True)
        self.h_scale_s = nn.Sequential(
            nn.TransformerEncoder(h_scale_s_layer, num_layers=2),
            nn.Linear(N, M),
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.M + (self.M // self.num_slices) * min(i, 5), 224),
                nn.GELU(),
                nn.Linear(224, 128),
                nn.GELU(),
                nn.Linear(128, (self.M // self.num_slices)),
            )
            for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.M + (self.M // self.num_slices) * min(i, 5), 224),
                nn.GELU(),
                nn.Linear(224, 128),
                nn.GELU(),
                nn.Linear(128, (self.M // self.num_slices)),
            )
            for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.M + (self.M // self.num_slices) * min(i + 1, 6), 224),
                nn.GELU(),
                nn.Linear(224, 128),
                nn.GELU(),
                nn.Linear(128, (self.M // self.num_slices)),
            )
            for i in range(self.num_slices)
        )

        # 숙제: entropy_bottleneck 만 써먹는 뉴럴 코텍 만들기
        # 조건: [b, s, 64] -> [b, 64, s] ->  [b, 64, s, 1] 같은 이상한 꼼수 쓰지 마샘
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[1:]
        z = self.h_a(y)

        # [b, s, 64] -> [b, 64, s]
        z = z.permute(0, 2, 1).contiguous()
        # import ipdb; ipdb.set_trace()
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()  # mean 대신 median을 사용?
        # print(z.shape, z_offset.shape, y.shape, z_likelihoods.shape)
        z_offset = z_offset.permute(1, 0, 2).contiguous()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        # [b, 64, s] -> [b, s, 64]
        z_hat = z_hat.permute(0, 2, 1).contiguous()

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, -1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[: self.max_support_slices]

            mean_support = torch.cat([latent_means] + support_slices, dim=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=-1)
            scale = self.cc_scale_transforms[slice_index](scale_support)

            # gaussian entropy 태우기 위해 4차원으로 변형 (트릭임)
            y_slice = y_slice.permute(0, 2, 1).contiguous()
            mu = mu.permute(0, 2, 1).contiguous()
            scale = scale.permute(0, 2, 1).contiguous()

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu  # Q(y−μ)+μ (μ = mean). ste_round가 round & encode 연산임.

            y_hat_slice = y_hat_slice.permute(0, 2, 1).contiguous()

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=-1)
        y_likelihoods = torch.cat(y_likelihood, dim=-1)
        x_hat = self.g_s(y_hat)

        return {"x": x, "x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[1:]

        z = self.h_a(y)

        z = z.permute(0, 2, 1).contiguous()

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-1:])

        z_hat = z_hat.permute(0, 2, 1).contiguous()

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, -1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[: self.max_support_slices]

            mean_support = torch.cat([latent_means] + support_slices, dim=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=-1)
            scale = self.cc_scale_transforms[slice_index](scale_support)

            # gaussian entropy 태우기 위해 4차원으로 변형 (트릭임)
            y_slice = y_slice.permute(0, 2, 1).contiguous()
            mu = mu.permute(0, 2, 1).contiguous()
            scale = scale.permute(0, 2, 1).contiguous()

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu  # Q(y−μ)+μ (μ = mean)

            y_hat_slice = y_hat_slice.permute(0, 2, 1).contiguous()

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        # y를 quantize한 뒤에 gaussian distribution을 가지고 bitstream으로 만들어버림
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-1:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        ## ryu
        # y_shape = [z_hat.shape[-2], z_hat.shape[-1]]
        y_shape = [z_hat.shape[-1]]

        z_hat = z_hat.permute(0, 2, 1).contiguous()

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = y_hat_slices if self.max_support_slices < 0 else y_hat_slices[: self.max_support_slices]

            mean_support = torch.cat([latent_means] + support_slices, dim=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=-1)
            scale = self.cc_scale_transforms[slice_index](scale_support)

            # gaussian entropy 태우기 위해 4차원으로 변형 (트릭임)
            mu = mu.permute(0, 2, 1).contiguous()
            scale = scale.permute(0, 2, 1).contiguous()

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            ## ryu
            # rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0])

            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            y_hat_slice = y_hat_slice.permute(0, 2, 1).contiguous()

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=-1)
        x_hat = self.g_s(y_hat)

        return {"x_hat": x_hat}
