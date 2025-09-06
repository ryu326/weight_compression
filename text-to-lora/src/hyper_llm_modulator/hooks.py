import logging
from collections import OrderedDict
from operator import attrgetter

import torch
import torch.nn.functional as F

from hyper_llm_modulator.utils import get_layers

logger = logging.getLogger()


def remove_hooks_(module):
    module._forward_hooks = OrderedDict()
    module._foward_pre_hooks = OrderedDict()


def remove_all_hooks_(model):
    layers = get_layers(model)
    for layer in layers:
        for module in layer.modules():
            remove_hooks_(module)


def remove_hook_handles_(handles):
    for handle in handles:
        if handle is not None:
            handle.remove()


def apply_hook_to_layer_(
    layer,
    mname,
    pre_hook=None,
    post_hook=None,
    remove_other_hooks=False,
):
    # assert mname in ["block", "mlp", "self_attn"]
    assert (pre_hook is not None) or (post_hook is not None)
    pre_hook_handle = None
    post_hook_handle = None
    if "block" == mname:
        module = layer
    else:
        if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
            mname = f"self_attn.{mname}"
        elif mname in ["gate_proj", "down_proj", "up_proj"]:
            mname = f"mlp.{mname}"
        # assert hasattr(layer, mname)
        module = attrgetter(mname)(layer)
    if remove_other_hooks:
        remove_hooks_(module)
    if pre_hook is not None:
        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
    if post_hook is not None:
        post_hook_handle = module.register_forward_hook(post_hook)
    return pre_hook_handle, post_hook_handle


def apply_steering_hooks_at_layers_(
    model,
    module_names,
    layer_indices,
    vec,
    alpha=1.0,
    remove_other_hooks=False,
):
    if "block" in module_names:
        assert len(module_names) == 1

    logger.info(f"Applying steering hooks to layers {layer_indices} with module names {module_names}")

    def add_vec_hook(module, args, output):
        newoutput = output[0] + alpha * vec
        return (newoutput, *output[1:])

    return apply_custom_hooks_at_layers_(
        model,
        module_names,
        layer_indices,
        post_hook=add_vec_hook,
        remove_other_hooks=remove_other_hooks,
    )


def apply_steering_hooks_all_layers_(
    model,
    module_names: list[str],
    vec: torch.Tensor,
    alpha=1.0,
    remove_other_hooks: bool = False,
):
    layer_indices = list(range(len(get_layers(model))))
    return apply_steering_hooks_at_layers_(
        model,
        module_names,
        layer_indices,
        vec,
        alpha,
        remove_other_hooks,
    )


def apply_custom_hooks_at_layers_(
    model,
    module_names,
    layer_indices,
    pre_hook=None,
    post_hook=None,
    remove_other_hooks=False,
):
    if "block" in module_names:
        assert len(module_names) == 1

    layers = get_layers(model)
    out_handles = []
    c = 0
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for mname in module_names:
            handles = apply_hook_to_layer_(
                layer,
                mname,
                pre_hook=pre_hook,
                post_hook=post_hook,
                remove_other_hooks=remove_other_hooks,
            )
            out_handles += handles
            if handles[0] is not None or handles[1] is not None:
                c += 1
    if c == 0:
        logger.warning("No forward hooks applied. Something might be wrong. Check if the module names are correct.")
    return out_handles


def add_last_layer_activation_hook(model, output_tensor):

    @torch.no_grad()
    def hook(module, input, output):
        output_tensor.copy_(output)

    return apply_custom_hooks_at_layers_(
        model,
        ["block"],
        [len(get_layers(model)) - 1],
        post_hook=hook,
    )


def add_lora_hooks(model, module_names, layer_indices, A, B, scaling, input_dropout, training):
    # A: [bs, in_features, r]
    # B: [bs, r, out_features]
    def lora_hook(module, args, output):
        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        x = args[0].to(A.dtype)
        bs, inp_len = x.shape[:2]

        # A and B repeat for each input token
        lora_A = A.repeat_interleave(inp_len, dim=0)
        lora_B = B.repeat_interleave(inp_len, dim=0)
        x = x.reshape(bs * inp_len, 1, -1)
        delta_x = torch.bmm(torch.bmm(F.dropout(x, input_dropout, training), lora_A), lora_B) * scaling

        newoutput = model_out + delta_x.reshape(bs, inp_len, -1).to(model_out.dtype)
        if isinstance(output, tuple):
            return (newoutput, *output[1:])
        else:
            return newoutput

    return apply_custom_hooks_at_layers_(
        model,
        module_names,
        layer_indices,
        post_hook=lora_hook,
    )


def add_vera_hooks(model, module_names, layer_indices, A, B, d, b, scaling, input_dropout):

    def vera_hook(module, args, output):
        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        x = args[0].to(A.dtype)
        bs, inp_len = x.shape[:2]

        vera_d = d.squeeze(-1).repeat_interleave(inp_len, dim=0)
        vera_b = b.squeeze(-1).repeat_interleave(inp_len, dim=0)
        in_features = x.shape[-1]
        out_features = vera_b.shape[-1]
        x = x.reshape(bs * inp_len, in_features)
        delta_x = F.linear(F.linear(F.dropout(x, input_dropout), A) * vera_d, B) * vera_b

        newoutput = model_out + delta_x.reshape(bs, inp_len, -1).to(model_out.dtype)
        if isinstance(output, tuple):
            return (newoutput, *output[1:])
        else:
            return newoutput

    return apply_custom_hooks_at_layers_(
        model,
        module_names,
        layer_indices,
        post_hook=vera_hook,
    )
