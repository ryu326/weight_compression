import torch
from torch import nn


def save_linear(module, path):
    saved_layer = torch.load(path, map_location=torch.device('cpu'))
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = (
        module.SV.data.float() /
        saved_layer['Wscale'].float().to(module.SV.data.device)).cpu()
    if module.tlut is not None:
        saved_layer['tlut'] = module.tlut.data.to(torch.float16)
    torch.save(saved_layer, path)


def _maybe_qwen3_rotary(layer, device):
    """Build a Qwen3RotaryEmbedding if `layer` is a Qwen3DecoderLayer.
    Newer transformers (>=4.51) require attention to receive
    position_embeddings=(cos,sin); older Llama did not."""
    cfg = getattr(getattr(layer, "self_attn", None), "config", None)
    if cfg is None:
        cfg = getattr(layer, "config", None)
    if getattr(cfg, "model_type", "") not in ("qwen3", "qwen3_moe"):
        return None
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    return Qwen3RotaryEmbedding(config=cfg, device=device)


def calculate_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    qwen3_rotary = _maybe_qwen3_rotary(layer, device)
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            inp = source.to(device)
            kwargs = {"position_ids": position_ids}
            if qwen3_rotary is not None:
                kwargs["position_embeddings"] = qwen3_rotary(inp, position_ids)
            total_loss += nn.MSELoss()(layer(inp, **kwargs)[0], target)
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_mse_loss_moe(layer, dataloader, device, attention_mask, rotary_emb): # 1. attention_mask 인자 추가
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    
    # 2. 어텐션 마스크를 한 번만 device로 이동
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            
            # 3. layer 호출 시 attention_mask 전달
            forward_kwargs = {
                "hidden_states": source.to(device),
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
            # [수정] Qwen용 Position Embeddings 계산 및 추가
            if rotary_emb is not None:
                position_embeddings = rotary_emb(source.to(device), position_ids)
                forward_kwargs["position_embeddings"] = position_embeddings

            output = layer(**forward_kwargs)[0]            
            # output = layer(source.to(device),
            #                position_ids=position_ids,
            #                attention_mask=attention_mask)[0]
                           
            total_loss += nn.MSELoss()(output, target)
            ct += 1
            
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_ce_loss_model(model, dataloader, start_dev, in_q, out_q):
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            in_q.put(source)
            output = model(source.to(start_dev))['logits'][:, :-1].contiguous()
            output = output.view(-1, output.shape[-1])
            target = out_q.get().to(output.device)
            target = target.view(-1, target.shape[-1])
            total_loss += nn.CrossEntropyLoss()(output, target)
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()

def calculate_mse_loss_quip(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            total_loss += nn.MSELoss()(layer(source.to(device), position_ids=position_ids)[0],
                                       target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()