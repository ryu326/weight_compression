# import argparse
# import os
# import time

# import glog
# import torch
# from transformers import AutoTokenizer

# from lib import codebook, utils
# from lib.utils.unsafe_import import model_from_hf_path
# from model.llama import LlamaForCausalLM
# from transformers import LlamaForCausalLM as OrigLlama
# import json

# torch.set_grad_enabled(False)

# parser = argparse.ArgumentParser()
# parser.add_argument('--quantized_path', type=str)
# parser.add_argument('--hf_output_path', type=str)
# parser.add_argument('--skip_list', type=str)
# parser.add_argument('--use_codes', action='store_true')
# parser.add_argument('--W_key', type=str, default='')



# def main(args):
#     assert os.path.exists(args.quantized_path)
#     saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
#     model_config = saved_config['model_config']
#     glog.info(model_config)
#     fused = model_config.quip_params.get('fused', True)

#     tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)
    
#     # model = OrigLlama.from_pretrained(model_config._name_or_path,
#     #                                             torch_dtype='auto',
#     #                                             low_cpu_mem_usage=True,
#     #                                             config=model_config)
#     model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
#                                                 torch_dtype='auto',
#                                                 low_cpu_mem_usage=True,
#                                                 config=model_config)


#     orig_model = OrigLlama.from_pretrained(model_config._name_or_path,
#                                            torch_dtype='auto',
#                                            low_cpu_mem_usage=True,
#                                            config=model_config)

#     print("--- 모델 생성 직후 메모리 주소 확인 ---")
#     # 에러 메시지에 나왔던 레이어 번호 두 개를 넣으세요 (예: 17, 31)
#     addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm)
#     addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm)
#     print(f"Layer 17 row_norm 주소: {addr1}")
#     print(f"Layer 31 row_norm 주소: {addr2}")
#     print(f"주소 동일 여부: {addr1 == addr2}")


#     # try:
#     #     if skip_list is None:
#     #         skip_list = []
#     # except:
#     #     skip_list = []
    
#     # skip_list = args.skip_list.split(',') if args.skip_list else []
#     skip_list = []
    
#     comp_result = {}
#     comp_result['bpp_loss'] = 0
#     comp_result['bpp'] = 0
#     comp_result['ppl'] = 0
#     comp_result['num_pixels'] = 0
    
#     try:
#         with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
#             saved_config = json.load(f)
#         comp_result['config'] = saved_config
#     except Exception as e:
#         print(f"Failed to load config: {e}")

    
#     cpu = torch.device('cpu')
#     if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
#         lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
#                                  map_location=cpu, weights_only=False)
#         model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
#             model.lm_head.weight.dtype))
#         model.model.norm.weight.copy_(lmhead_data['norm'].to(
#             model.model.norm.weight.dtype))

#     for ii in range(len(model.model.layers)):
#         layer = model.model.layers[ii]

#         if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
#             ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
#                                  map_location=cpu, weights_only=False)
#             layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
#                 layer.input_layernorm.weight.dtype))
#             layer.post_attention_layernorm.weight.copy_(
#                 ln_data['post_attention_layernorm'].to(
#                     layer.post_attention_layernorm.weight.dtype))

#         if f'{ii}_q' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_q.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             # print(rnorm.shape)
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.self_attn.q_proj.weight.copy_(W_hat.to(layer.self_attn.q_proj.weight.dtype))
#             layer.self_attn.q_proj.row_norm.copy_(rnorm.to(layer.self_attn.q_proj.weight.dtype))
            
#             comp_result[f'{ii}_q.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.self_attn.q_proj = orig_model.model.layers[ii].self_attn.q_proj
        
#         if f'{ii}_k' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_k.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             # print(rnorm.shape)
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.self_attn.k_proj.weight.copy_(W_hat.to(layer.self_attn.k_proj.weight.dtype))
#             layer.self_attn.k_proj.row_norm.copy_(rnorm.to(layer.self_attn.k_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.self_attn.k_proj.weight.copy_(W_hat.to(layer.self_attn.k_proj.weight.dtype))            
#             comp_result[f'{ii}_k.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.self_attn.k_proj = orig_model.model.layers[ii].self_attn.k_proj
            
#         if f'{ii}_v' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_v.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             # print(rnorm.shape)
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.self_attn.v_proj.weight.copy_(W_hat.to(layer.self_attn.v_proj.weight.dtype))
#             layer.self_attn.v_proj.row_norm.copy_(rnorm.to(layer.self_attn.v_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.self_attn.v_proj.weight.copy_(W_hat.to(layer.self_attn.v_proj.weight.dtype))            
#             comp_result[f'{ii}_v.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.self_attn.v_proj = orig_model.model.layers[ii].self_attn.v_proj

#         if f'{ii}_o' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.self_attn.o_proj.weight.copy_(W_hat.to(layer.self_attn.o_proj.weight.dtype))
#             layer.self_attn.o_proj.row_norm.copy_(rnorm.to(layer.self_attn.o_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.self_attn.o_proj.weight.copy_(W_hat.to(layer.self_attn.o_proj.weight.dtype))            
#             comp_result[f'{ii}_o.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.self_attn.o_proj = orig_model.model.layers[ii].self_attn.o_proj

#         if f'{ii}_up' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.mlp.up_proj.weight.copy_(W_hat.to(layer.mlp.up_proj.weight.dtype))
#             layer.mlp.up_proj.row_norm.copy_(rnorm.to(layer.mlp.up_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.mlp.up_proj.weight.copy_(W_hat.to(layer.mlp.up_proj.weight.dtype))            
#             comp_result[f'{ii}_up.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.mlp.up_proj = orig_model.model.layers[ii].mlp.up_proj
            
#         if f'{ii}_gate' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_gate.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.mlp.gate_proj.weight.copy_(W_hat.to(layer.mlp.gate_proj.weight.dtype))
#             layer.mlp.gate_proj.row_norm.copy_(rnorm.to(layer.mlp.gate_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.mlp.gate_proj.weight.copy_(W_hat.to(layer.mlp.gate_proj.weight.dtype))            
#             comp_result[f'{ii}_gate.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.mlp.gate_proj = orig_model.model.layers[ii].mlp.gate_proj
                      
#         if f'{ii}_down' not in skip_list:
#             saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
#                                      map_location=cpu, weights_only=False)
#             rnorm = saved_layer['row_norm']
#             W_hat = saved_layer['W_hat' + args.W_key] / rnorm
#             layer.mlp.down_proj.weight.copy_(W_hat.to(layer.mlp.down_proj.weight.dtype))
#             layer.mlp.down_proj.row_norm.copy_(rnorm.to(layer.mlp.down_proj.weight.dtype))
            
#             # W_hat = saved_layer['W_hat' + args.W_key]
#             # layer.mlp.down_proj.weight.copy_(W_hat.to(layer.mlp.down_proj.weight.dtype))            
#             comp_result[f'{ii}_down.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
#             comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
#             comp_result['num_pixels'] += saved_layer['num_pixels']
#             comp_result['bpp'] += saved_layer['bpp_sum']
#         else:
#             print('### skip ####')
#             layer.mlp.down_proj = orig_model.model.layers[ii].mlp.down_proj

#         glog.info(f'loaded layer {ii}')
        
        
#     print("\n--- 가중치 로딩 후 메모리 주소 확인 ---")
#     addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm)
#     addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm)
#     print(f"Layer 17 row_norm 주소: {addr1}")
#     print(f"Layer 31 row_norm 주소: {addr2}")
#     print(f"주소 동일 여부: {addr1 == addr2}")

    
#     comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
#     comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']    
    
#     if type(comp_result['bpp_loss']) == torch.Tensor:
#         comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
#     if type(comp_result['bpp']) == torch.Tensor:
#         comp_result['bpp'] = comp_result['bpp'].item()

#     glog.info(f'saving model...')
#     model.save_pretrained(args.hf_output_path, safe_serialization=True)
#     tokenizer.save_pretrained(args.hf_output_path)
#     del model   
    
#     file_path = f'{args.hf_output_path}_result'
#     if os.path.exists(file_path):
#         # file_path = f'{args.hf_output_path}_result2.json'
#         os.rename(file_path + '.json', file_path + '_.json')
#     with open(file_path + '.json', 'w') as f:
#         json.dump(comp_result, f, indent=2)
        
#     # model, _ = model_from_hf_path(args.hf_output_path, device_map='cuda')

#     # glog.info('successfully loaded hfized model')
#     # glog.info('generating some text...')

#     # start = time.time()
#     # prompt = 'It is a truth universally acknowledged that'
#     # inputs = tokenizer(prompt, return_tensors='pt')
#     # outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
#     #                          attention_mask=inputs['attention_mask'].cuda(),
#     #                          max_new_tokens=64,
#     #                          return_dict_in_generate=True)
#     # token = outputs.sequences[0, :]
#     # output_str = tokenizer.decode(token)
#     # glog.info(output_str)
#     # glog.info(f'elapsed: {time.time() - start}')

# if __name__ == '__main__':
#     torch.set_grad_enabled(False)
#     torch.manual_seed(0)
#     args = parser.parse_args()
#     main(args)
import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from transformers import LlamaForCausalLM as OrigLlama
import json

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--skip_list', type=str, default='') # 기본값을 빈 문자열로 설정
parser.add_argument('--use_codes', action='store_true')
parser.add_argument('--W_key', type=str, default='')

def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']
    glog.info(model_config)
    fused = model_config.quip_params.get('fused', True)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)
    
    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                                torch_dtype='auto',
                                                low_cpu_mem_usage=True,
                                                config=model_config)


    orig_model = OrigLlama.from_pretrained(model_config._name_or_path,
                                           torch_dtype='auto',
                                           low_cpu_mem_usage=True,
                                           config=model_config)

    print("--- 모델 생성 직후 메모리 주소 확인 ---")
    addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm.data)
    addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm.data)
    print(f"Layer 27 o_proj.row_norm.data 주소: {addr1}")
    print(f"Layer 29 q_proj.row_norm.data 주소: {addr2}")
    print(f"주소 동일 여부: {addr1 == addr2}")


    skip_list = args.skip_list.split(',') if args.skip_list else []
    
    comp_result = {}
    comp_result['bpp_loss'] = 0
    comp_result['bpp'] = 0
    comp_result['ppl'] = 0
    comp_result['num_pixels'] = 0
    
    try:
        with open(os.path.join(args.quantized_path, 'config.json'), 'r') as f:
            saved_config = json.load(f)
        comp_result['config'] = saved_config
    except Exception as e:
        print(f"Failed to load config: {e}")

    
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu, weights_only=False)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
            model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(
            model.model.norm.weight.dtype))

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]
        print(f"\n===== Layer {ii} =====")

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu, weights_only=False)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        if f'{ii}_q' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_q.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  q_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.self_attn.q_proj.weight.copy_(W_hat.to(layer.self_attn.q_proj.weight.dtype))
            layer.self_attn.q_proj.row_norm.copy_(rnorm.to(layer.self_attn.q_proj.weight.dtype))
            
            comp_result[f'{ii}_q.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_q ####')
            layer.self_attn.q_proj = orig_model.model.layers[ii].self_attn.q_proj
        
        if f'{ii}_k' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_k.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  k_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.self_attn.k_proj.weight.copy_(W_hat.to(layer.self_attn.k_proj.weight.dtype))
            layer.self_attn.k_proj.row_norm.copy_(rnorm.to(layer.self_attn.k_proj.weight.dtype))
            
            comp_result[f'{ii}_k.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_k ####')
            layer.self_attn.k_proj = orig_model.model.layers[ii].self_attn.k_proj
            
        if f'{ii}_v' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_v.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  v_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.self_attn.v_proj.weight.copy_(W_hat.to(layer.self_attn.v_proj.weight.dtype))
            layer.self_attn.v_proj.row_norm.copy_(rnorm.to(layer.self_attn.v_proj.weight.dtype))
            
            comp_result[f'{ii}_v.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_v ####')
            layer.self_attn.v_proj = orig_model.model.layers[ii].self_attn.v_proj

        if f'{ii}_o' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  o_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.self_attn.o_proj.weight.copy_(W_hat.to(layer.self_attn.o_proj.weight.dtype))
            layer.self_attn.o_proj.row_norm.copy_(rnorm.to(layer.self_attn.o_proj.weight.dtype))
            
            comp_result[f'{ii}_o.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_o ####')
            layer.self_attn.o_proj = orig_model.model.layers[ii].self_attn.o_proj

        if f'{ii}_up' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  up_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.mlp.up_proj.weight.copy_(W_hat.to(layer.mlp.up_proj.weight.dtype))
            layer.mlp.up_proj.row_norm.copy_(rnorm.to(layer.mlp.up_proj.weight.dtype))
            
            comp_result[f'{ii}_up.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_up ####')
            layer.mlp.up_proj = orig_model.model.layers[ii].mlp.up_proj
            
        if f'{ii}_gate' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_gate.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  gate_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.mlp.gate_proj.weight.copy_(W_hat.to(layer.mlp.gate_proj.weight.dtype))
            layer.mlp.gate_proj.row_norm.copy_(rnorm.to(layer.mlp.gate_proj.weight.dtype))
            
            comp_result[f'{ii}_gate.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_gate ####')
            layer.mlp.gate_proj = orig_model.model.layers[ii].mlp.gate_proj
                      
        if f'{ii}_down' not in skip_list:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
                                     map_location=cpu, weights_only=False)
            rnorm = saved_layer['row_norm']
            ### 추가된 코드 ###
            print(f"  down_proj: rnorm shape={rnorm.shape}, id={id(rnorm.data)}")
            ##################
            W_hat = saved_layer['W_hat' + args.W_key] / rnorm
            layer.mlp.down_proj.weight.copy_(W_hat.to(layer.mlp.down_proj.weight.dtype))
            layer.mlp.down_proj.row_norm.copy_(rnorm.to(layer.mlp.down_proj.weight.dtype))
            
            comp_result[f'{ii}_down.pt'] = {k:v for k, v in saved_layer.items() if not isinstance(v, torch.Tensor) and k != 'codes'}
            comp_result['bpp_loss'] += saved_layer['bpp_loss_sum' + args.W_key]
            comp_result['num_pixels'] += saved_layer['num_pixels']
            comp_result['bpp'] += saved_layer['bpp_sum']
        else:
            print(f'### skip {ii}_down ####')
            layer.mlp.down_proj = orig_model.model.layers[ii].mlp.down_proj
        
        
    print("\n--- 가중치 로딩 후 메모리 주소 확인 ---")
    addr1 = id(model.model.layers[27].self_attn.o_proj.row_norm.data)
    addr2 = id(model.model.layers[29].self_attn.q_proj.row_norm.data)
    print(f"Layer 27 o_proj.row_norm.data 주소: {addr1}")
    print(f"Layer 29 q_proj.row_norm.data 주소: {addr2}")
    print(f"주소 동일 여부: {addr1 == addr2}")

    
    comp_result['bpp_loss'] = comp_result['bpp_loss'] / comp_result['num_pixels']
    comp_result['bpp'] = comp_result['bpp'] / comp_result['num_pixels']    
    
    if type(comp_result['bpp_loss']) == torch.Tensor:
        comp_result['bpp_loss'] = comp_result['bpp_loss'].item()
    if type(comp_result['bpp']) == torch.Tensor:
        comp_result['bpp'] = comp_result['bpp'].item()

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.hf_output_path)
    del model   
    
    file_path = f'{args.hf_output_path}_result'
    if os.path.exists(file_path):
        os.rename(file_path + '.json', file_path + '_.json')
    with open(file_path + '.json', 'w') as f:
        json.dump(comp_result, f, indent=2)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)