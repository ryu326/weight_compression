import torch
import math
# import utils
from lib import utils
import os
from lib.algo import quip

def compress_linear(W, H, comp_model, ql, args, device='cpu'):
    
    comp_model = comp_model.to(device)
    comp_model.scale = comp_model.scale.to(device)
    comp_model.shift = comp_model.shift.to(device)
    W = W.to(device)
    H = H.to(device)
    ql = ql.to(device) if ql is not None else None
    
    SU, SV = None, None
    if args.incoh_mode != 'none':
        Lhr, H, W, SU, SV, scaleWH = quip.incoherence_preprocess(H, W, args) 
        
    if args.gptq:
        out = pseudo_compress_tensor_gptq(W, comp_model, args.direction, args.comp_batch_size, ql, H, device, args)                
    elif args.ldlq:
        out = pseudo_compress_tensor_ldlq(W, comp_model, args.direction, args.comp_batch_size, ql, H, device, args)
    else:
        out = pseudo_compress_tensor(W, comp_model, args.direction, args.comp_batch_size, ql, None, device, args)   

    if args.incoh_mode != 'none':
        out['w_hat'] = quip.incoherence_process(out['w_hat'], SU, SV, scaleWH, args)
        
    utils.clean()
    return out['w_hat'], out['bpp_loss_sum'], out['num_pixels'], SU, SV, scaleWH

def compress_weight_block_with_model(weight_block, model, ql=None):
    
    ori_shape = weight_block.shape
    if ori_shape[-1] != model.input_size:
        weight_block = weight_block.reshape(ori_shape[0], -1, model.input_size)
        
    data = {}
    data['weight_block'] = weight_block
    if ql is not None:
        # assert ql.shape[0] == ori_shape
        data['q_level'] = ql.reshape(ori_shape[0],)
    # import ipdb; ipdb.set_trace()
    # print(data['q_level'].shape)
    out = model(data)
    w_hat = out['x_hat'].reshape(ori_shape)
    
    num_pixels = weight_block.numel()
    if isinstance(out["likelihoods"], dict):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / -math.log(2))
            for likelihoods in out["likelihoods"].values()
        ).item()
    else :
        bpp_loss = (torch.log(out["likelihoods"]).sum() / -math.log(2)).item()

        # out_enc = model.compress(data)
    # try:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"], data["q_level"])
    # except:
    #     out_dec = model.decompress(out_enc["strings"][0], out_enc["shape"])
    
    # for s in out_enc["strings"]:
    #     bpp += len(s[0]) * 8.0

    return w_hat, num_pixels, bpp_loss
    
def pseudo_compress_tensor(w, model, direction, bs, q_level, hess_eigen, device, args):
    if direction == 'col':
        w = w.T
    ori_shape = w.shape
    
    if args.bundle:
        w = w.reshape(ori_shape[0], -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
    else :
        w = w.reshape(-1, model.input_size)
        
    w_hat = torch.zeros(w.shape, dtype=w.dtype, device=w.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    if q_level is not None:
        q_level = q_level.reshape(-1, )

    if hess_eigen is not None:
        R = hess_eigen['eigenvectors'].size(0)
        hess_eigen = hess_eigen['eigenvectors'].to(device)
        hess_eigen = hess_eigen.reshape(1, R, -1, model.input_size).repeat(bs, 1, 1, 1)

    for start_idx in range(0, w.shape[0], bs):
        end_idx = min(start_idx + bs, w.shape[0])
        batch = w[start_idx:end_idx]
        
        ql = None
        if q_level is not None:
            ql = q_level[start_idx:end_idx]

        x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(batch, model, ql)
        
        w_hat[start_idx:end_idx] = x_hat
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    w_hat = w_hat.reshape(ori_shape).cpu()

    if direction == 'col':
        w_hat = w_hat.T
    
    return {'w_hat': w_hat,
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}

def pseudo_compress_tensor_gptq(W, model, direction, bs, q_level, H, device, args):
    
    assert direction == 'col'
    if q_level is not None:
        q_level = q_level.reshape(-1, )
    
    W = W.to(device)
    Losses = torch.zeros_like(W, device=W.device)
    Q = torch.zeros_like(W, device=W.device)
    assert torch.isfinite(H).all()
    
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    assert torch.isfinite(H).all()
    
    rows = W.shape[0]
    columns = W.shape[1]
    
    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for i1 in range(0, columns, bs):
        i2 = min(i1 + bs, columns)
        count = i2 - i1
        
        ql = None
        if q_level is not None:
            ql = q_level[i1:i2]

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1, device=W1.device)
        Err1 = torch.zeros_like(W1, device=W1.device)
        Losses1 = torch.zeros_like(W1, device=W1.device)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            
            assert w.size(-1) == rows
            if args.bundle:
                w_reshape = w.reshape(1, -1, model.input_size)  # (row, col) --> (row, -1, inputsize)
            else :
                w_reshape = w.reshape(-1, model.input_size)

            ql_ = None if ql is None else ql[i]
            x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(w_reshape, model, ql)
            
            q = x_hat.flatten()
            num_pixels += n_pixels
            bpp_loss += bpp_loss_

            Q1[:, i] = q
            Losses1[:, i] = (w - q)**2 / d**2

            err1 = (w - q) / d
            assert torch.isfinite(err1).all()
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return {'w_hat': Q.cpu(),
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}

def pseudo_compress_tensor_ldlq(Wr, model, direction, bs, q_level, H, device, args):
    assert direction == 'col'
    
    L, D = block_LDL(H, 128)
    
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros_like(Wr, dtype=Wr.dtype, device=Wr.device)

    num_pixels = 0
    bpp_loss = 0
    bpp = 0

    for k in reversed(range(n)):
        WXWX = Wr[:, k:k+1] + (Wr[:, k+1:] - hatWr[:, k+1:]) @ L[k+1:, k:k+1]
    
        ql = None if q_level is None else q_level[k]
        x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql)
        
        q = x_hat.flatten()
        hatWr[:, k] = q
        num_pixels += n_pixels
        bpp_loss += bpp_loss_

    for ie in range(args.quip_tune_iters):
        raise Exception
        num_pixels = 0
        bpp_loss = 0
        bpp = 0
        
        for k in reversed(range(n)):
            WXWX = hatWr[:, k:k+1] + (Wr - hatWr) @ H[:, k:k+1] @ torch.linalg.inv(H[k:k+1, k:k+1])
            
            ql_ = None if q_level is None else q_level[i]
            x_hat, n_pixels, bpp_loss_ = compress_weight_block_with_model(WXWX.reshape(1, -1, model.input_size), model, ql)
            
            q = x_hat.flatten()
            hatWr[:, k] = q
            
            if ie == args.quip_tune_iters - 1:
                num_pixels += n_pixels
                bpp_loss += bpp_loss_

    return {'w_hat': hatWr.cpu(),
            'bpp_loss_sum': bpp_loss,
            'num_pixels': num_pixels,
            'bpp': bpp}


def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    try:
        L = torch.linalg.cholesky(H)
    except:
        return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))