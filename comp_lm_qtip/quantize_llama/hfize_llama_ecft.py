import argparse
import json
import os
from operator import attrgetter
from typing import Any

import glog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.linear.ec_linear import EntropyConstrainedLinear

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser()
parser.add_argument("--quantized_path", type=str, required=True)
parser.add_argument("--hf_output_path", type=str, required=True)
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--skip_list", type=str, default="")
parser.add_argument("--ec_decoder_type", type=str, default=None, choices=["rht", "dft", "identity"])


def _is_gemma3_config(config):
    return getattr(config, "model_type", "") in ("gemma3", "gemma3_text")


def _get_text_model(model):
    return model.language_model if hasattr(model, "language_model") else model.model


def _as_float(value, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _resolve_decoder_type(comp_config, args) -> str:
    if args.ec_decoder_type is not None:
        return args.ec_decoder_type
    if isinstance(comp_config, dict):
        return comp_config.get("ec_decoder_type", "rht")
    return getattr(comp_config, "ec_decoder_type", "rht")


def _projection_specs():
    return [
        ("q", "self_attn.q_proj"),
        ("k", "self_attn.k_proj"),
        ("v", "self_attn.v_proj"),
        ("o", "self_attn.o_proj"),
        ("up", "mlp.up_proj"),
        ("gate", "mlp.gate_proj"),
        ("down", "mlp.down_proj"),
    ]


def _extract_substate(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    key_prefix = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(key_prefix):
            out[key[len(key_prefix):]] = value
    return out


def _copy_layernorms_from_state(layer, state_dict: dict[str, Any]) -> int:
    copied = 0
    for norm_name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        if not hasattr(layer, norm_name):
            continue
        key = f"{norm_name}.weight"
        if key not in state_dict:
            continue
        target = getattr(layer, norm_name).weight
        target.copy_(state_dict[key].to(dtype=target.dtype))
        copied += 1
    return copied


def _apply_decoder_state_to_layer(
    *,
    layer,
    layer_idx: int,
    decoder_state: dict[str, Any],
    decoder_type: str,
    skip_set: set[str],
) -> dict[str, Any]:
    result = {
        "ec_loaded": 0,
        "dense_loaded": 0,
        "skipped": 0,
        "missing": [],
        "norms_loaded": 0,
    }
    result["norms_loaded"] = _copy_layernorms_from_state(layer, decoder_state)

    for short_name, path in _projection_specs():
        skip_key = f"{layer_idx}_{short_name}"
        if skip_key in skip_set:
            result["skipped"] += 1
            continue

        target_linear = attrgetter(path)(layer)
        latent_key = f"{path}.latent"
        weight_key = f"{path}.weight"

        if latent_key in decoder_state:
            sub = _extract_substate(decoder_state, path)
            latent = sub["latent"]
            out_features, in_features = latent.shape
            ec_layer = EntropyConstrainedLinear(
                in_features=in_features,
                out_features=out_features,
                bias=(target_linear.bias is not None),
                decoder_type=decoder_type,
                device=latent.device,
                dtype=latent.dtype,
            )
            ec_layer.load_state_dict(sub, strict=False)
            ec_layer.eval()
            with torch.no_grad():
                weight, _ = ec_layer.reconstruct_weight(training=False, quantized=True)
            target_linear.weight.copy_(weight.to(dtype=target_linear.weight.dtype))
            if target_linear.bias is not None and "bias" in sub and sub["bias"] is not None:
                target_linear.bias.copy_(sub["bias"].to(dtype=target_linear.bias.dtype))
            result["ec_loaded"] += 1
            continue

        if weight_key in decoder_state:
            target_linear.weight.copy_(decoder_state[weight_key].to(dtype=target_linear.weight.dtype))
            bias_key = f"{path}.bias"
            if target_linear.bias is not None and bias_key in decoder_state:
                target_linear.bias.copy_(decoder_state[bias_key].to(dtype=target_linear.bias.dtype))
            result["dense_loaded"] += 1
            continue

        result["missing"].append(path)

    return result


def _load_optional_lmhead(model, orig_model, quantized_path: str) -> None:
    text_model = _get_text_model(model)
    orig_text_model = _get_text_model(orig_model)
    cpu = torch.device("cpu")
    lmhead_path = os.path.join(quantized_path, "lmhead.pt")

    if os.path.exists(lmhead_path):
        lmhead_data = torch.load(lmhead_path, map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data["lm_head"].to(model.lm_head.weight.dtype))
        text_model.norm.weight.copy_(lmhead_data["norm"].to(text_model.norm.weight.dtype))
        return

    assert torch.equal(model.lm_head.weight, orig_model.lm_head.weight), "LM heads do not match!"
    assert torch.equal(text_model.norm.weight, orig_text_model.norm.weight), "Final norms do not match!"
    assert torch.equal(
        text_model.embed_tokens.weight, orig_text_model.embed_tokens.weight
    ), "Embeddings do not match!"


def _accumulate_rate_stats(comp_result: dict[str, Any], quantized_path: str, layer_idx: int) -> None:
    for short_name, _ in _projection_specs():
        path = os.path.join(quantized_path, f"{layer_idx}_{short_name}.pt")
        if not os.path.exists(path):
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        comp_result["bpp_loss_sum"] += _as_float(saved.get("bpp_loss_sum", 0.0))
        comp_result["bpp_sum"] += _as_float(saved.get("bpp_sum", 0.0))
        comp_result["num_pixels"] += _as_float(saved.get("num_pixels", 0.0))
        comp_result["metadata_total_raw_sum"] += _as_float(saved.get("metadata_total_raw", 0.0))
        comp_result["decoder_total_raw_sum"] += _as_float(saved.get("decoder_total_raw", 0.0))


def main(args):
    if not os.path.exists(args.quantized_path):
        raise FileNotFoundError(args.quantized_path)

    config_path = os.path.join(args.quantized_path, "config.pt")
    saved_config = torch.load(config_path, weights_only=False)
    model_config = saved_config["model_config"]
    comp_config = saved_config["quant_args"]
    decoder_type = _resolve_decoder_type(comp_config, args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    is_gemma3 = _is_gemma3_config(model_config)
    if is_gemma3:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            config=model_config,
        )
        orig_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            config=model_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            config=model_config,
        )
        orig_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            config=model_config,
        )

    _load_optional_lmhead(model, orig_model, args.quantized_path)
    text_model = _get_text_model(model)

    skip_set = set(x.strip() for x in args.skip_list.split(",") if x.strip())
    glog.info(f"decoder_type={decoder_type}, skip_list={sorted(skip_set)}")

    comp_result = {
        "loaded_decoder_layers": 0,
        "total_layers": len(text_model.layers),
        "decoder_type": decoder_type,
        "bpp_loss_sum": 0.0,
        "bpp_sum": 0.0,
        "num_pixels": 0.0,
        "metadata_total_raw_sum": 0.0,
        "decoder_total_raw_sum": 0.0,
        "bpp_loss": 0.0,
        "bpp": 0.0,
        "metadata_total": 0.0,
        "bpp_with_metadata": 0.0,
        "bpp_loss_with_metadata": 0.0,
        "decoder_total": 0.0,
        "layers": {},
    }

    config_json_path = os.path.join(args.quantized_path, "config.json")
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r") as f:
                comp_result["config"] = json.load(f)
        except Exception as exc:
            glog.warning(f"failed to load config.json: {exc}")

    for ii, layer in enumerate(text_model.layers):
        decoder_ft_path = os.path.join(args.quantized_path, f"{ii}_decoder_ft.pt")
        if str(ii) in skip_set or f"{ii}_decoder" in skip_set:
            glog.info(f"skip layer {ii} by skip_list")
            continue
        if not os.path.exists(decoder_ft_path):
            glog.warning(f"decoder file missing for layer {ii}: {decoder_ft_path}")
            continue

        dec = torch.load(decoder_ft_path, map_location="cpu", weights_only=False)
        decoder_state = dec.get("state_dict", dec)
        if not isinstance(decoder_state, dict):
            glog.warning(f"invalid decoder state format at layer {ii}")
            continue

        layer_info = _apply_decoder_state_to_layer(
            layer=layer,
            layer_idx=ii,
            decoder_state=decoder_state,
            decoder_type=decoder_type,
            skip_set=skip_set,
        )
        _accumulate_rate_stats(comp_result, args.quantized_path, ii)
        comp_result["layers"][str(ii)] = layer_info
        comp_result["loaded_decoder_layers"] += 1
        glog.info(
            f"loaded decoder {ii}: ec={layer_info['ec_loaded']} dense={layer_info['dense_loaded']} "
            f"skip={layer_info['skipped']} missing={len(layer_info['missing'])}"
        )

    if comp_result["num_pixels"] > 0:
        comp_result["bpp_loss"] = comp_result["bpp_loss_sum"] / comp_result["num_pixels"]
        comp_result["bpp"] = comp_result["bpp_sum"] / comp_result["num_pixels"]
        comp_result["metadata_total"] = (
            comp_result["metadata_total_raw_sum"] / comp_result["num_pixels"]
        )
        comp_result["bpp_with_metadata"] = (
            comp_result["bpp_sum"] + comp_result["metadata_total_raw_sum"]
        ) / comp_result["num_pixels"]
        comp_result["bpp_loss_with_metadata"] = (
            comp_result["bpp_loss_sum"] + comp_result["metadata_total_raw_sum"]
        ) / comp_result["num_pixels"]
        comp_result["decoder_total"] = (
            comp_result["decoder_total_raw_sum"] / comp_result["num_pixels"]
        )

    glog.info("saving model...")
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)

    out_json = f"{args.hf_output_path}_result.json"
    if os.path.exists(out_json):
        os.rename(out_json, f"{args.hf_output_path}_result_.json")
    with open(out_json, "w") as f:
        json.dump(comp_result, f, indent=2)
    glog.info(f"saved result json: {out_json}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main(parser.parse_args())

