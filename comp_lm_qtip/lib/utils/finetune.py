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


def calculate_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            total_loss += nn.MSELoss()(layer(source.to(device),
                                             position_ids=position_ids)[0],
                                       target)
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


import torch
import torch.nn as nn

def get_initial_weights(layer, dataloader, device):
    # print("INFO: Calculating initial loss weights for warm-up...")
    layer.eval()  # 평가 모드로 전환
    source, target = next(iter(dataloader))

    with torch.no_grad():
        source = source.to(device)
        target = target.to(device)
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)

        output = layer(source, position_ids=position_ids)[0]
        l_mse = nn.MSELoss()(output, target)
        l_bpp = sum(m.bpp_loss for m in layer.modules() if hasattr(m, 'bpp_loss'))

        epsilon = 1e-8
        
        # w_mse = l_bpp / (l_mse + l_bpp)
        # w_bpp = l_mse / (l_mse + l_bpp)
        w_mse = l_bpp / (l_mse + l_bpp + epsilon)
        w_bpp = l_mse / (l_mse + l_bpp + epsilon)
        
        initial_w = torch.tensor([w_mse, w_bpp], device=device)
    # print(f"Initial losses -> MSE: {l_mse.item():.4f}, BPP: {l_bpp.item():.2f}")
    # print(f"Calculated initial weights -> MSE: {w_mse:.3f}, BPP: {w_bpp:.3f}")
    
    layer.train()  # 다시 학습 모드로 전환
    return initial_w


def calculate_test_loss(layer, dataloader, device):
    layer.eval()  # 모델을 평가 모드로 설정

    total_mse = 0.0
    total_bpp = 0.0
    ct = 0
    position_ids = None
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = layer(source, position_ids=position_ids)[0]
            mse_loss = nn.MSELoss()(output, target)
            bpp_loss = sum(m.bpp_loss for m in layer.modules()
                           if hasattr(m, 'bpp_loss')) # 'bpp_loss' 속성을 가진 모든 모듈을 찾음
            total_mse += mse_loss
            total_bpp += bpp_loss
            ct += 1
            
    avg_mse = (total_mse / ct).cpu().item()
    avg_bpp = (total_bpp / ct).cpu().item()
    layer.train()
    return avg_mse, avg_bpp