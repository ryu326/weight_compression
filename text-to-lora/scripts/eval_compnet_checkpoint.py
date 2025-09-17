import sys
from argparse import ArgumentParser
# from hyper_llm_modulator.sft_trainer import eval_hypermod_checkpoint
from hyper_llm_modulator.utils.eval_compnet import eval_compnet_checkpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--full_eval", action="store_true", default=False)
    parser.add_argument("--use_icl", action="store_true", default= False)
    # eval_compnet_checkpoint(**vars(parser.parse_args()), curstep=None)
    
    args = parser.parse_args()
    eval_compnet_checkpoint(args.checkpoint_path, args.device, '_final', full_eval=args.full_eval, use_icl = args.use_icl)
    
    # eval_compnet_checkpoint(**vars(parser.parse_args()), curstep=None)
