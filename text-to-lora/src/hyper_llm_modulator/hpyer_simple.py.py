import math
from math import sqrt
from typing import Dict, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists


# ---- helpers ----

class TaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, encoded_task_emb_size: int):
        super().__init__()
        self.encoded_task_emb_size = encoded_task_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"encoded_task_emb": self.mlp(x)}


class MLPResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


def get_in_out_features(model: PeftModel, peft_config: Optional[PeftConfig] = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    in_features, out_features = {}, {}
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        assert isinstance(module.base_layer, nn.Linear)
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules
        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features
    return in_features, out_features


def get_init_peft_weights(model: PeftModel, peft_config: Optional[PeftConfig] = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        assert isinstance(module.base_layer, nn.Linear)
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules
        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in submodule:
                continue
            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                # same type check
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)
    return peft_weights


def zero_lora_param_dict(target_modules, n_layers, rank, in_features, out_features):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(n_layers, rank, in_features[m]), requires_grad=False)
                 for m in target_modules}
            ),
            "B": nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(n_layers, out_features[m], rank), requires_grad=False)
                 for m in target_modules}
            ),
        }
    )


# ---- the simplified HyperModulator (shared_AB_head=True, autoreg_gen=False, LoRA only) ----

class HyperModulatorSharedAB(nn.Module):
    def __init__(
        self,
        model: PeftModel,
        module_names: Dict[str, List[List[str]]],  # target_module -> per-layer list of ["...lora_A...", "...lora_B..."]
        *,
        training_task: Literal["sft", "recon"] = "sft",
        task_emb_size: Optional[int] = None,
        head_in_size: int = 512,
        latent_size: int = 128,
        learnable_AB_offset: bool = False,   # only meaningful for SFT (offset added)
        mt_AB_offset: Optional[dict] = None, # tensor-dict with shapes matching zero_lora_param_dict
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert training_task in ["sft", "recon"]
        self.model_config = model.config
        self.peft_config: PeftConfig = model.peft_config["default"]
        self.output_space = "lora"
        self.shared_AB_head = True
        self.autoreg_gen = False
        self.training_task = training_task
        self.device = model.device
        self.dtype = dtype

        # LoRA scaling (note: 실제 적용은 LoRA 모듈 쪽이 담당하는 경우가 많음)
        self.scaling = self.peft_config.lora_alpha / self.peft_config.r
        if getattr(self.peft_config, "use_rslora", False):
            self.scaling *= math.sqrt(self.peft_config.r)

        # 모듈 입출력 차원 맵
        self.in_features, self.out_features = get_in_out_features(model, self.peft_config)
        self.target_modules = self.peft_config.target_modules
        self.module_names = module_names
        self.max_num_layers = self.model_config.num_hidden_layers

        # (선택) 태스크 임베딩
        encoded_task_emb_size = 0
        if task_emb_size is not None:
            encoded_task_emb_size = latent_size // 2
            self.task_encoder = TaskEncoder(task_emb_size, encoded_task_emb_size).to(self.device)

        # 레이어/타입 임베딩 (연속형)
        depth_emb_size = latent_size // 4
        type_emb_size  = latent_size // 4
        self.layer_depth_encoder = nn.Sequential(
            nn.Embedding(self.max_num_layers, depth_emb_size),
            nn.LayerNorm(depth_emb_size),
        )
        self.layer_type_encoder = nn.Sequential(
            nn.Embedding(len(self.target_modules), type_emb_size),
            nn.LayerNorm(type_emb_size),
        )
        self.module_to_int = {m: i for i, m in enumerate(self.target_modules)}

        mlp_inp_size = depth_emb_size + type_emb_size + encoded_task_emb_size

        # trunk
        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.mlp1 = MLPResidualBlock(mlp_inp_size, mlp_inp_size * 4, mlp_inp_size, pre_layer_norm=True, post_dropout=True)
        self.mlp2 = MLPResidualBlock(mlp_inp_size, mlp_inp_size * 4, mlp_inp_size, pre_layer_norm=True, post_dropout=True)
        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        # shared A/B head: 모듈별 A/B 임베딩 벡터 + 단일 head
        self.out_emb = nn.ParameterDict(
            (
                m,
                nn.ParameterDict(
                    dict(
                        A=nn.Parameter(torch.normal(0, 1, size=(mlp_inp_size,)), requires_grad=True),
                        B=nn.Parameter(torch.normal(0, 1, size=(mlp_inp_size,)), requires_grad=True),
                    )
                ),
            )
            for m in self.target_modules
        )
        self.out_emb_norm = nn.LayerNorm(mlp_inp_size)

        # 모듈별 shared head: r * max(in, out)
        heads = []
        for m in self.target_modules:
            output_size = self.peft_config.r * max(self.in_features[m], self.out_features[m])
            heads.append((m, nn.Linear(head_in_size, output_size, bias=False, device=self.device)))
        self.heads = nn.ModuleDict(heads)

        # 초기 PEFT 가중치에서 bias-HyperInit 유사 초기화
        peft_weights = get_init_peft_weights(model, self.peft_config)
        self.split_shapes = {}
        with torch.no_grad():
            for m, head in self.heads.items():
                A0 = peft_weights[m]["lora_A"].weight.clone().flatten()
                B0 = peft_weights[m]["lora_B"].weight.clone().flatten()
                init_bias_like = A0 / sqrt(2)   # shared에서는 A로 스케일 맞추는 전략 사용
                if head.bias is None:
                    head.bias = nn.Parameter(torch.zeros(head.out_features, device=head.weight.device))
                # bias 길이가 더 작을 수 있으니 슬라이스
                size = head.bias.shape[0]
                head.bias.copy_(init_bias_like[:size])
                self.split_shapes[m] = [len(A0), len(B0)]

        # (옵션) SFT일 때 A/B 오프셋
        self.AB_offset = zero_lora_param_dict(
            self.target_modules, self.max_num_layers, self.peft_config.r,
            self.in_features, self.out_features
        )
        if (training_task == "sft") and (mt_AB_offset is not None):
            # 주입된 tensor dict을 파라미터로 교체
            for which in ["A", "B"]:
                for m in self.target_modules:
                    self.AB_offset[which][m] = nn.Parameter(
                        mt_AB_offset[which][m], requires_grad=learnable_AB_offset
                    )

        self.to(self.device).to(self.dtype)

    # ---- embeddings ----

    def _embed_layer_depth(self, depth_indices):
        if isinstance(depth_indices, int):
            depth_indices = torch.tensor([depth_indices], dtype=torch.long, device=self.device)
        elif isinstance(depth_indices, list):
            depth_indices = torch.tensor(depth_indices, dtype=torch.long, device=self.device)
        return self.layer_depth_encoder(depth_indices)

    def _embed_layer_type(self, layer_type: str):
        idx = self.module_to_int[layer_type]
        idx = torch.tensor([idx], dtype=torch.long, device=self.device)
        return self.layer_type_encoder(idx)

    # ---- core forward for one layer type (shared_AB_head, non-autoreg) ----

    def _hypernet_forward(self, layer_indices: torch.Tensor, layer_type: str,
                          encoded_task_emb: Optional[torch.Tensor]):
        bs = len(layer_indices)

        depth_emb = self._embed_layer_depth(layer_indices)                 # [bs, d_depth]
        type_emb  = self._embed_layer_type(layer_type).expand(bs, -1)     # [bs, d_type]
        if encoded_task_emb is None:
            encoded_task_emb = torch.empty((bs, 0), device=self.device)

        cat = torch.cat([encoded_task_emb, depth_emb, type_emb], dim=-1)  # [bs, d_total]
        h = self.mixer(cat)
        h = self.mlp1(h)

        # 공통 trunk 끝
        head = self.heads[layer_type]
        in_feat  = self.in_features[layer_type]
        out_feat = self.out_features[layer_type]
        r = self.peft_config.r

        splitted_out = []
        # A 경로
        hA = self.mlp2(h + self.out_emb_norm(self.out_emb[layer_type]["A"]))
        hA = self.mlp3(hA)
        yA = head(hA)                               # [bs, r*max(in,out)]
        yA = yA.view(bs, r, -1)[..., :in_feat]      # [bs, r, in]
        A_flat = yA.reshape(bs, r * in_feat)
        splitted_out.append(A_flat)

        # B 경로
        hB = self.mlp2(h + self.out_emb_norm(self.out_emb[layer_type]["B"]))
        hB = self.mlp3(hB)
        yB = head(hB)                               # [bs, r*max(in,out)]
        yB = yB.view(bs, r, -1)[..., :out_feat]     # [bs, r, out]
        B_flat = yB.reshape(bs, r * out_feat)
        splitted_out.append(B_flat)

        return splitted_out  # [A_flat, B_flat]

    # ---- public APIs ----

    def get_delta_weights(self, layer_indices: torch.Tensor, layer_type: str,
                          encoded_task_emb: Optional[torch.Tensor] = None,
                          *, factorized: bool = False):
        bs = len(layer_indices)
        A_flat, B_flat = self._hypernet_forward(layer_indices, layer_type, encoded_task_emb)

        r = self.peft_config.r
        in_feat  = self.in_features[layer_type]
        out_feat = self.out_features[layer_type]

        A = A_flat.reshape(bs, r, in_feat)                 # (bs, r, in)
        B = B_flat.reshape(bs, r, out_feat).transpose(-1, -2)  # (bs, out, r)

        if self.training_task == "sft":
            A = A + self.AB_offset["A"][layer_type][layer_indices]
            B = B + self.AB_offset["B"][layer_type][layer_indices]

        if factorized:
            return A, B
        deltaW = torch.bmm(B, A)                           # (bs, out, in)
        return deltaW

    @torch.no_grad()
    def gen_lora(self, layer_indices: torch.Tensor, encoded_task_emb: torch.Tensor):
        """ layer_indices: (L,), encoded_task_emb: (1, d_task_enc) 또는 (L, d_task_enc) """
        assert encoded_task_emb.ndim == 2
        if encoded_task_emb.shape[0] == 1:
            encoded_task_emb = encoded_task_emb.expand(layer_indices.shape[0], -1)

        lora_A, lora_B = {}, {}
        for m in self.target_modules:
            A, B = self.get_delta_weights(layer_indices, m, encoded_task_emb, factorized=True)
            lora_A[m], lora_B[m] = A, B

        # LoRA state_dict 포맷으로 변환 (module_names에 따라 이름 매핑)
        lora_sd = {}
        for m in self.target_modules:
            for layer_idx in layer_indices:
                for name in self.module_names[m][layer_idx]:
                    if "lora_A" in name:
                        lora_sd[name] = lora_A[m][layer_idx].contiguous().cpu()
                    elif "lora_B" in name:
                        lora_sd[name] = lora_B[m][layer_idx].contiguous().cpu()
                    else:
                        raise ValueError(f"Unexpected module name: {name}")
        return lora_sd
