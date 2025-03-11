import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glog
import gc

def clean():
    gc.collect()
    torch.cuda.empty_cache()

def get_susv_adam(susv_params, params, args):
    return torch.optim.Adam([
        {
            'params': susv_params,
            'lr': args.ft_susv_lr
        },
        {
            'params': params,
            'lr': args.ft_lr
        },
    ])

def get_adam(params, args):
    return torch.optim.Adam([
        {
            'params': params,
            'lr': args.ft_lr
        },
    ])

def extract_susv_params(module):
    susv_params = []
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'SU' in name or 'SV' in name:
                susv_params.append(param)
            else:
                params.append(param)
    return susv_params, params


def extract_ft_params(module):
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            params.append(param)
    return params

def calculate_mse_loss(layer, dataloader, device):
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

class SimpleDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def split_data(X, Y, args):
    split = int(len(X) - args.ft_valid_size)
    glog.info(f'using {split} training seqs, {len(X) - split} validation seqs')
    train_ds = SimpleDataset(X[:split], Y[:split])
    valid_ds = SimpleDataset(X[split:], Y[split:])
    train_dl = DataLoader(train_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=True)
    valid_dl = DataLoader(valid_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=False)
    return train_dl, valid_dl