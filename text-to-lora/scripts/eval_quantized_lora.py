import sys
from argparse import ArgumentParser
# from hyper_llm_modulator.sft_trainer import eval_hypermod_checkpoint
from hyper_llm_modulator.utils.eval_quantization import eval_quantized_lora

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument("--use-icl", action="store_true")
    parser.add_argument(
        "--group", 
        type=int, 
        default=128, 
    )
    parser.add_argument(
        "--bit", 
        type=int, 
        default=None, 
        help="Quantization bits for LoRA matrices (e.g., 2, 4, 8)."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="channel", 
        choices=['tensor', 'channel', 'group'], 
        help="Quantization mode ('tensor', 'channel', or 'group')."
    )
    args = parser.parse_args()
    
    if args.bit != None:
        bits = [args.bit]
    else:
        bits = range(2,9)
    
    for bit in bits:
        print(f"############ eval quant {args.mode} {args.group} bit{bit} ################")
        quant_cfg = {
            "A": {
                "mode": args.mode,
                "bits": bit,
                "group_size": args.group,
            },
            "B": {
                "mode": args.mode,
                "bits": bit,
                "group_size": args.group,
            },
        }
        eval_quantized_lora(
            save_dir=args.save_dir,
            device=args.device,
            full_eval=args.full_eval,
            use_icl=args.use_icl,
            quant_cfg=quant_cfg, 
            curstep=None
        )