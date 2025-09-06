export CUDA_VISIBLE_DEVICES=2,3
uv run python scripts/eval_compnet_checkpoint.py \
    --checkpoint_path /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/hyper_lora/20250904-101519_axGMqysg/checkpoints/latest_it270000.pt \
    --full_eval

export CUDA_VISIBLE_DEVICES=3
uv run python scripts/eval_compnet_checkpoint.py \
    --checkpoint_path /workspace/Weight_compression/text-to-lora/train_outputs/compnet_recon/hyper_lora/20250904-101518_5rcL6bSr/checkpoints/latest_it470000.pt \
    --full_eval



# import sys
# from argparse import ArgumentParser
# # from hyper_llm_modulator.sft_trainer import eval_hypermod_checkpoint
# from hyper_llm_modulator.utils.eval_compnet import eval_compnet_checkpoint

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=str, required=True)
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--full_eval", action="store_true")
#     parser.add_argument("--use-icl", action="store_true")
#     eval_compnet_checkpoint(**vars(parser.parse_args()), curstep=None)
