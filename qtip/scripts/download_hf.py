import argparse

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str)
parser.add_argument('--repo_id', type=str)
parser.add_argument('--read_token', type=str)
"doesn't save to huggingface cache, only downloads in specified local directory"
if __name__ == "__main__":
    args = parser.parse_args()
    snapshot_download(repo_id=args.repo_id,
                      repo_type="model",
                      local_dir=args.folder_path,
                      local_dir_use_symlinks=False,
                      token=args.read_token)


# python scripts/download_hf.py --folder_path /workspace/Weight_compression/Wparam_dataset/quip_hess/llama2_70b_relaxml --repo_id relaxml/Hessians-Llama-2-70b-6144 --read_token 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