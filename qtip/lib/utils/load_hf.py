import torch
import torch.nn as nn
import copy
import accelerate
import transformers
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig

# =============================================================================
# [V1] Classes: Individual Linear Layer + ModuleList Approach
# =============================================================================

try:
    from model.llama import LlamaForCausalLM
except:
    LlamaForCausalLM = None

def model_from_hf_path_gptoss(path, max_mem_ratio=0.7, device_map=None, sep_rnorm=False, gptoss_replace_version=None):
    model_cls = transformers.AutoModelForCausalLM if not sep_rnorm else LlamaForCausalLM
    model_str = path

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        try:
            model = model_cls.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True, attn_implementation='sdpa', trust_remote_code=True)
        except:
            model = model_cls.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True, trust_remote_code=True)
        device_map = accelerate.infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'], max_memory=mmap)
        
    try:
        model = model_cls.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True, attn_implementation='sdpa', device_map=device_map, trust_remote_code=True)
    except:
        model = model_cls.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device_map, trust_remote_code=True)

    if 'gptoss' in path or 'gpt-oss' in path:     
        if gptoss_replace_version == 'v1':
            model = replace_gptoss_modules_v1(model)
        elif gptoss_replace_version == 'v1.1':
            model = replace_gptoss_modules_v1(model, v11 = True)
        elif gptoss_replace_version == 'v2':
            model = replace_gptoss_modules_v2(model)
        elif gptoss_replace_version == 'v3':
            model = replace_gptoss_modules_v3(model)
        else:
            model = model
            
    return model, model_str


class GptOssLayerV1(nn.Module):
    def __init__(self, config, alpha=1.702, limit=7.0):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        
        self.alpha = alpha
        self.limit = limit

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        
        if self.limit is not None and self.limit > 0:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        
        glu = gate * torch.sigmoid(gate * self.alpha)
        return self.down_proj((up + 1) * glu)

class GptOssExpertsV1(nn.Module):
    def __init__(self, config, alpha=1.702, limit=7.0):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        # 개별 Linear Expert 생성
        self.experts = nn.ModuleList([
            GptOssLayerV1(config, alpha, limit) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hitted[:]:
            idx_int = expert_idx.item()
            expert_layer = self.experts[idx_int]

            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx[0]])
            
            current_state = hidden_states[token_idx]
            expert_output = expert_layer(current_state)
            
            weighted_output = expert_output * routing_weights[token_idx, idx_int].unsqueeze(-1)
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

class GptOssExpertsV11(nn.Module):
    def __init__(self, config, alpha=1.702, limit=7.0):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        # 개별 Linear Expert 생성
        self.experts = nn.ModuleList([
            GptOssLayerV1(config, alpha, limit) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            # expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # for expert_idx in expert_hitted[:]:
        for idx_int in range(num_experts):
            # idx_int = expert_idx.item()
            expert_layer = self.experts[idx_int]

            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[idx_int])
            
            current_state = hidden_states[token_idx]
            expert_output = expert_layer(current_state)
            
            if token_idx.numel() > 0:
                weighted_output = expert_output * routing_weights[token_idx, idx_int].unsqueeze(-1)
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

class GptOssTopKRouterV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.gate(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices 


# =============================================================================
# [V2] Classes: ModuleList of Linears (Separate Up/Down)
# =============================================================================

class GptOssTopKRouterV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.gate(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices

class GptOssExpertsV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        
        self.gate_up_experts = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=True)
            for _ in range(self.num_experts)
        ])
        self.down_experts = nn.ModuleList([
            nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
            for _ in range(self.num_experts)
        ])
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
            
        if True: # Always use sparse path
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=num_experts + 1
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            
            for expert_idx in expert_hit[:]:
                expert_idx = expert_idx[0]
                if expert_idx == num_experts:
                    continue
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                
                idx_int = expert_idx.item()
                gate_up_layer = self.gate_up_experts[idx_int]
                down_layer = self.down_experts[idx_int]
                gate_up = gate_up_layer(current_state)
                
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = down_layer(gated_output)
                
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
            
        else:
            raise
        return next_states


# =============================================================================
# [V3] Classes: 3D Parameter Tensor (Robust Logic)
# =============================================================================

class GptOssExpertsV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        
        if True: # Force sparse execution
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=num_experts + 1
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            
            for expert_idx in expert_hit[:]:
                expert_idx = expert_idx[0]
                if expert_idx == num_experts:
                    continue
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            raise
        return next_states


# =============================================================================
# Helper Functions (Model Loading & Replacement)
# =============================================================================


def replace_gptoss_modules_v1(model, v11 = False):
    print("🔄 Standard Module V1 교체 작업 시작...")
    config = model.config
    
    for i, layer in enumerate(model.model.layers):
        old_router = layer.mlp.router
        old_experts = layer.mlp.experts
        
        orig_alpha = getattr(old_experts, 'alpha', 1.702)
        orig_limit = getattr(old_experts, 'limit', 7.0)
        
        new_router = GptOssTopKRouterV1(config)
        new_router.to(old_router.weight.device).to(old_router.weight.dtype)
        new_router.gate.weight.data.copy_(old_router.weight.data)
        if old_router.bias is not None:
            new_router.gate.bias.data.copy_(old_router.bias.data)
            
        if v11:
            new_experts = GptOssExpertsV11(config, alpha=orig_alpha, limit=orig_limit)
        else:
            new_experts = GptOssExpertsV1(config, alpha=orig_alpha, limit=orig_limit)
        new_experts.to(old_experts.gate_up_proj.device).to(old_experts.gate_up_proj.dtype)

        old_gate_up = old_experts.gate_up_proj.data 
        old_gate_up_bias = old_experts.gate_up_proj_bias.data 
        old_down = old_experts.down_proj.data 
        old_down_bias = old_experts.down_proj_bias.data 
        
        for idx in range(config.num_local_experts):
            target_expert = new_experts.experts[idx]
            
            w_gate = old_gate_up[idx] 
            b_gate = old_gate_up_bias[idx]
            target_expert.gate_up_proj.weight.data.copy_(w_gate.t())
            target_expert.gate_up_proj.bias.data.copy_(b_gate)
            
            w_down = old_down[idx]
            b_down = old_down_bias[idx]
            target_expert.down_proj.weight.data.copy_(w_down.t())
            target_expert.down_proj.bias.data.copy_(b_down)

        layer.mlp.router = new_router
        layer.mlp.experts = new_experts
        
        if i == 0:
            print(f"   ℹ️ Layer 0 변환 완료 (Alpha: {orig_alpha}, Limit: {orig_limit})")

    print("✅ 모든 레이어의 V1 모듈 교체가 완료되었습니다!")
    return model

def replace_gptoss_modules_v2(model):
    print("🔄 Standard Module V2 교체 작업 시작...")
    config = model.config
    
    for i, layer in enumerate(model.model.layers):
        old_router = layer.mlp.router
        old_experts = layer.mlp.experts
        
        orig_alpha = getattr(old_experts, 'alpha', 1.702)
        orig_limit = getattr(old_experts, 'limit', 7.0)
        
        new_router = GptOssTopKRouterV2(config)
        new_router.to(old_router.weight.device).to(old_router.weight.dtype)
        new_router.gate.weight.data.copy_(old_router.weight.data)
        if old_router.bias is not None:
            new_router.gate.bias.data.copy_(old_router.bias.data)
            
        new_experts = GptOssExpertsV2(config)
        new_experts.alpha = orig_alpha
        new_experts.limit = orig_limit
        new_experts.to(old_experts.gate_up_proj.device).to(old_experts.gate_up_proj.dtype)

        old_gate_up = old_experts.gate_up_proj.data 
        old_gate_up_bias = old_experts.gate_up_proj_bias.data 
        old_down = old_experts.down_proj.data 
        old_down_bias = old_experts.down_proj_bias.data 
        
        for idx in range(config.num_local_experts):
            w_gate = old_gate_up[idx] 
            b_gate = old_gate_up_bias[idx]
            new_experts.gate_up_experts[idx].weight.data.copy_(w_gate.t())
            new_experts.gate_up_experts[idx].bias.data.copy_(b_gate)
            
            w_down = old_down[idx]
            b_down = old_down_bias[idx]
            new_experts.down_experts[idx].weight.data.copy_(w_down.t())
            new_experts.down_experts[idx].bias.data.copy_(b_down)

        layer.mlp.router = new_router
        layer.mlp.experts = new_experts
        
        if i == 0:
            print(f"   ℹ️ Layer 0 변환 완료 (Alpha: {orig_alpha}, Limit: {orig_limit})")

    print("✅ 모든 레이어의 V2 모듈 교체가 완료되었습니다!")
    return model


def replace_gptoss_modules_v3(model):
    print("🔄 Standard Module V3 교체 작업 시작...")
    config = model.config
    
    for i, layer in enumerate(model.model.layers):
        old_experts = layer.mlp.experts
        
        orig_alpha = getattr(old_experts, 'alpha', 1.702)
        orig_limit = getattr(old_experts, 'limit', 7.0)
        
        new_experts = GptOssExpertsV3(config)
        new_experts.to(old_experts.gate_up_proj.device).to(old_experts.gate_up_proj.dtype)

        old_gate_up = old_experts.gate_up_proj.data
        old_gate_up_bias = old_experts.gate_up_proj_bias.data 
        old_down = old_experts.down_proj.data
        old_down_bias = old_experts.down_proj_bias.data
         
        new_experts.gate_up_proj.data.copy_(old_gate_up)
        new_experts.gate_up_proj_bias.data.copy_(old_gate_up_bias)
        new_experts.down_proj.data.copy_(old_down)
        new_experts.down_proj_bias.data.copy_(old_down_bias)
        
        layer.mlp.experts = new_experts
        
        if i == 0:
            print(f"   ℹ️ Layer 0 변환 완료 (Alpha: {orig_alpha}, Limit: {orig_limit})")

    print("✅ 모든 레이어의 V3 모듈 교체가 완료되었습니다!")
    return model