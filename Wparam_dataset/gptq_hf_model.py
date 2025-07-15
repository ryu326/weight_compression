import torch
from transformers import CLIPVisionModelWithProjection, ViTForImageClassification, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer, GPTQConfig
from tqdm import tqdm


model_list = [
    # 'meta-llama/Llama-2-7b-hf',
    # 'meta-llama/Llama-2-13b-hf',
    'meta-llama/Meta-Llama-3-8B',
]

cache_directory = "./hf_model/cache"

for model_name in tqdm(model_list):
    print(model_name)
    
    save_directory = f"./hf_model/{model_name.replace('/', '--')}"  # 저장할 경로
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    
    for bit in [2, 3, 4, 8]:
        gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(save_directory, device_map="auto",  max_memory={0: "40GiB", 1: "40GiB", 2: "40GiB", 3: "40GiB", "cpu": "80GiB"}, quantization_config=gptq_config)
        quantized_model.to("cpu")
        quantized_model.save_pretrained(save_directory + f"_gptq_{bit}bit")
