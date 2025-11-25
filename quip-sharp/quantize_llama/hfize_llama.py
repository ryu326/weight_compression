import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from lib.utils.model_version import MODEL_VERSION

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'), weights_only=False)
    model_config = saved_config['model_config']

    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']
    # import ipdb; ipdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model_config.quip_params['model_version'] = MODEL_VERSION
    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config).half()
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'])
        model.model.norm.weight.copy_(lmhead_data['norm'])

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'])
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'])

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_qkv.pt',
                                 map_location=cpu)
        for i in range(len(saved_layer['scales'])):
            layer.self_attn.qkv_proj.fuse_scales[i].copy_(
                saved_layer['scales'][i])
        utils.unpack_quip(layer.self_attn.qkv_proj, saved_layer, codebook_id,
                          codesz)

        target_layer = layer.self_attn.qkv_proj
        
        # 1. 측정을 위해 임시로 GPU로 이동 (QuIP 커널은 CUDA에서 동작하므로 필수)
        device = torch.device("cuda")
        target_layer.to(device)
        
        # 2. 전체 가중치를 복원하기 위한 단위 행렬(Identity Matrix) 생성
        #    입력 차원(in_features) 크기의 단위 행렬을 넣으면 전체 가중치가 출력됨
        in_feat = target_layer.in_features
        dummy_input = torch.eye(in_feat, dtype=torch.float16, device=device)
        
        # 3. Warmup (초기화 오버헤드 제거 및 커널 컴파일)
        #    작은 배치로 한 번 실행하여 codebook_class 빌드 등을 트리거
        with torch.no_grad():
            _ = target_layer(dummy_input[:32, :])
        torch.cuda.synchronize()

        # 4. 시간 측정
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            # W_hat * I = W_hat (전체 매트릭스 디코딩 효과)
            _ = target_layer(dummy_input)
        end_event.record()
        
        torch.cuda.synchronize()
        
        latency_ms = start_event.elapsed_time(end_event)
        glog.info(f"[Layer {ii}] QKV W_hat Decoding Latency: {latency_ms:.4f} ms")

        # 5. 메모리 정리 및 CPU 원복 (메인 로직 흐름 유지)
        del dummy_input
        target_layer.to(cpu)
        torch.cuda.empty_cache()

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id,
                          codesz)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
                                 map_location=cpu)
        for i in range(len(saved_layer['scales'])):
            layer.mlp.upgate_proj.fuse_scales[i].copy_(
                saved_layer['scales'][i])
        utils.unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id,
                          codesz)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.mlp.down_proj, saved_layer, codebook_id,
                          codesz)
        glog.info(f'loaded layer {ii} down')

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path, use_cuda_graph=False)

    glog.info('successfully loaded hfized model')

    # glog.info('generating some text...')

    # start = time.time()
    # prompt = 'It is a truth universally acknowledged that'
    # inputs = tokenizer(prompt, return_tensors='pt')
    # outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
    #                          attention_mask=inputs['attention_mask'].cuda(),
    #                          max_new_tokens=64,
    #                          return_dict_in_generate=True)
    # token = outputs.sequences[0, :]
    # output_str = tokenizer.decode(token)
    # glog.info(output_str)
    # glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
