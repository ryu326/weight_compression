import argparse
import os

import tinychat.utils.constants
import torch
from accelerate import load_checkpoint_and_dispatch
from PIL import Image
# from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.models.vila_llama import VilaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator
from tinychat.utils.conversation_utils import (TimeStats, gen_params,
                                               stream_output)
from tinychat.utils.llava_image_processing import (load_images, process_images,
                                                   vis_images)
from tinychat.utils.prompt_templates import (get_image_token, get_prompter,
                                             get_stop_token_ids)
from tinychat.utils.tune import (device_warmup, tune_all_wqlinears,
                                 tune_llava_patch_embedding)
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def image_parser(args):
    out = args.image_file.split(args.im_sep)
    return out


def skip(*args, **kwargs):
    pass


def main(args):
    # Accelerate model initialization
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "llm"), use_fast=False)
    tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = tokenizer.convert_tokens_to_ids(
        [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = VilaLlamaForCausalLM(config).half()
    tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = tokenizer.convert_tokens_to_ids(
        [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    image_processor = vision_tower.image_processor
    # vision_tower = vision_tower.half()

    if args.precision == "W16A16":
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for i in pbar:
            model.llm = load_checkpoint_and_dispatch(
                model.llm,
                os.path.join(args.model_path, "llm"),
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                    "CLIPEncoderLayer",
                ],
            ).to(args.device)
        model = model.to(args.device)

    elif args.precision == "W4A16":
        from tinychat.utils.load_quant import load_awq_model

        model.llm = load_awq_model(model.llm, args.quant_path, 4, 128, args.device)
        from tinychat.modules import (make_fused_mlp, make_fused_vision_attn,
                                      make_quant_attn, make_quant_norm)

        if args.flash_attn:
            print("Enabling flash-attention!")
            make_quant_attn(model.llm, args.device, 1)
        else:
            print("Disabling flash-attention!")
            make_quant_attn(model.llm, args.device)
        make_quant_norm(model.llm)
        # make_fused_mlp(model)
        # make_fused_vision_attn(model,args.device)
        model = model.to(args.device)

    else:
        raise NotImplementedError(f"Precision {args.precision} is not supported.")

    image_files = image_parser(args)
    image_num = len(image_files)
    images = load_images(image_files)
    if args.vis_image:
        print("=" * 50)
        print("Input Image:")
        vis_images(image_files)
    # Similar operation in model_worker.py
    image_tensor = process_images(images, image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(args.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(args.device, dtype=torch.float16)

    device_warmup(args.device)
    tune_llava_patch_embedding(vision_tower, device=args.device)

    stream_generator = LlavaStreamGenerator

    if args.max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter(args.model_type, args.model_path, short_prompt, args.empty_prompt)
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    count = 0

    if args.empty_prompt:
        input_indicator = "Input: "
        output_indicator = "Generated: "
    else:
        input_indicator = "USER: "
        output_indicator = "ASSISTANT: "

    model.eval()
    time_stats = TimeStats()
    start_pos = 0
    while True:
        # Get input from the user
        print("=" * 50)
        input_prompt = input(input_indicator)
        print("-" * 50)
        if input_prompt == "":
            print("EXIT...")
            time_stats.show()
            break
        if count == 0:  # Insert image here
            image_token = get_image_token(model, args.model_path)
            image_token_holder = tinychat.utils.constants.LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER
            im_token_count = input_prompt.count(image_token_holder)
            if im_token_count == 0:
                model_prompter.insert_prompt(image_token * image_num + input_prompt)
            else:
                assert im_token_count == image_num
                input_prompt = input_prompt.replace(image_token_holder, image_token)
                model_prompter.insert_prompt(input_prompt)
        else:
            model_prompter.insert_prompt(input_prompt)
            if args.chunk_prefilling:
                image_tensor = None  # Can insert more images in future
        output_stream = stream_generator(
            model,
            tokenizer,
            model_prompter.model_input,
            start_pos,
            gen_params,
            device=args.device,
            stop_token_ids=stop_token_ids,
            image_tensor=image_tensor,
            chunk_prefilling=args.chunk_prefilling,
        )
        print(output_indicator, end="", flush=True)
        if count == 0:
            outputs, total_tokens = stream_output(output_stream, time_stats)
        else:
            outputs, total_tokens = stream_output(output_stream)
        if args.chunk_prefilling:
            start_pos += total_tokens
        if (
            args.single_round is not True and args.max_seq_len > 512
        ):  # Only memorize previous conversations when kv_cache_size > 512
            model_prompter.update_template(outputs, args.chunk_prefilling)
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="LLaMa", help="type of the model")
    parser.add_argument("--model-path", type=str, default="/data/llm/checkpoints/llava/llava-v1.5-7b")
    parser.add_argument(
        "--quant-path",
        type=str,
        default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt",
    )
    parser.add_argument("--precision", type=str, default="W4A16", help="compute precision")
    parser.add_argument(
        "--image-file",
        type=str,
        default="https://llava.hliu.cc/file=/nobackup/haotian/code/LLaVA/llava/serve/examples/extreme_ironing.jpg",
    )
    parser.add_argument(
        "--im-sep",
        type=str,
        default=",",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--single_round",
        action="store_true",
        help="whether to memorize previous conversations",
    )
    parser.add_argument(
        "--vis-image",
        action="store_true",
        help="whether to visualize the image while chatting",
    )
    parser.add_argument(
        "--empty-prompt",
        action="store_true",
        help="whether to use empty prompt template",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="whether to use flash attention",
    )
    parser.add_argument(
        "--chunk_prefilling",
        action="store_true",
        help="If used, in context stage, the history tokens will not be recalculated, greatly speeding up the calculation",
    )
    args = parser.parse_args()
    main(args)
