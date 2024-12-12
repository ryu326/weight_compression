import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math
import argparse
from tqdm import tqdm
import os

def calculate_perplexity(model, tokenizer, dataset, max_length=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in tqdm(dataset):
            inputs = tokenizer(example["text"], return_tensors="pt", max_length=max_length, truncation=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)  # Scale by number of tokens
            total_tokens += inputs["input_ids"].size(1)
            
            print('loss: ', loss, 'tokens: ', total_loss, 'Temp perplexity: ', math.exp(total_loss / total_tokens))

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    return perplexity

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a model's perplexity using WikiText-2 dataset.")
    parser.add_argument('--reconed_state_dict', type=str, default = None, help="Path to the reconstructed state dictionary file.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--save_path', type=str, default="./test_model_acc_results")
    return parser

# CUDA_VISIBLE_DEVICES=2 python test_model_acc.py --reconed_state_dict /home/jgryu/Weight_compression/nic_weight_comp/reconstruncted_state_dict/meta-llama--Meta-Llama-3-8B_mlp_d256_256.pth

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    # model_name = args.model_name
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="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", cache_dir='./model_cache_slurm')

    ckpt_path = '/home/jgryu/Weight_compression/llm-awq/model_cache/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920'
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, local_files_only=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Load reconstructed state dict
    if args.reconed_state_dict is not None:
        print('######### Test reconstructed Model!!')
        reconed_state_dict_path = args.reconed_state_dict
        print(f"Loading reconstructed state dict from: {reconed_state_dict_path}")
        reconed_state_dict = torch.load(reconed_state_dict_path)
        model.load_state_dict(reconed_state_dict)
    
    # Load WikiText dataset
    print('#### Dataset Loading')
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # Calculate Perplexity
    print('#### Start Testing')
    perplexity = calculate_perplexity(model, tokenizer, dataset)
    print(f"Model Perplexity on WikiText-2 Test Set: {perplexity}")

    results = {
        "model_name": args.model_name,
        "reconed_state_dict_path": args.reconed_state_dict,
        "perplexity": perplexity
    }
    
    os.makedirs(args.save_path, exist_ok = True)
    with open(os.path.join(args.save_path, f"{args.model_name}_{args.reconed_state_dict}.json"), "w") as json_file:
        json.dump(results, json_file, indent=4)
    print("Results saved to results.json")

if __name__ == "__main__":
    main()

    
