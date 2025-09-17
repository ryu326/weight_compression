# coding=utf-8
# ... (이전 import 구문들은 그대로 유지) ...
from accelerate import Accelerator

log: Logger = utils.get_logger("spinquant")

def train() -> None:
    accelerator = Accelerator()

    model_args, training_args, ptq_args = process_args_ptq()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
        
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    # 모델 로딩과 수정은 모든 프로세스가 각자 수행합니다.
    # 어차피 accelerator.prepare 단계에서 동기화되고 모델이 GPU에 분산됩니다.
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = ptq_model(ptq_args, model, model_args)
    
    # 모델 준비는 '단체 행동'이므로 모든 프로세스가 호출해야 합니다.
    model = accelerator.prepare(model)
    
    model.seqlen = 2048

    # Tokenizer는 메인 프로세스에서만 로드해도 되지만, 모든 프로세스가 가져도 무방합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_args.input_model)            
            
    datasets = ['wikitext2', 'c4']
    seed = 0
    seqlen = 2048

    # ==================== 수정된 로직 시작 ====================

    for dataset in datasets:
        # 데이터 로딩은 메인 프로세스에서만 수행하여 중복을 피합니다.
        input_tok = None
        if accelerator.is_main_process:
            try:
                input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                            seed=seed,
                                                            seqlen=seqlen,
                                                            model=model_args.input_model)
            except Exception as e:
                print(f"Error loading dataset {dataset}: {e}")
                input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                            seed=seed,
                                                            seqlen=seqlen,
                                                            model="../Wparam_dataset/hf_model/meta-llama--Meta-Llama-3-8B")
            nsamples = input_tok.numel() // seqlen
            input_tok = input_tok[0, :(seqlen * nsamples)].view(
                nsamples, seqlen)

        # 각 프로세스가 nsamples 정보를 동기화하도록 합니다.
        # 이 부분은 torch.distributed를 사용하거나, 간단하게 객체를 브로드캐스트해야 합니다.
        # 쉬운 방법: nsamples 값을 모든 프로세스에 전달
        if accelerator.is_main_process:
            nsamples_list = [input_tok.shape[0]]
        else:
            nsamples_list = [0]
        
        # main_process에서 다른 프로세스로 nsamples 값을 브로드캐스트
        accelerator.wait_for_everyone()
        torch.distributed.broadcast_object_list(nsamples_list, src=0)
        accelerator.wait_for_everyone()
        nsamples = nsamples_list[0]

        loss_fct = torch.nn.CrossEntropyLoss()
        acc_loss = 0.0
        
        # tqdm은 메인 프로세스에서만 보이도록 설정
        progress = tqdm(range(nsamples), disable=not accelerator.is_main_process)
        
        with torch.no_grad():
            for ii in progress:
                # 메인 프로세스에서 현재 스텝의 데이터를 가져옵니다.
                if accelerator.is_main_process:
                    batch = input_tok[ii, :].to(accelerator.device).view(1, -1)
                else:
                    # 다른 프로세스는 해당 shape의 빈 텐서를 준비합니다.
                    batch = torch.empty((1, seqlen), dtype=torch.long, device=accelerator.device)
                
                # 메인 프로세스가 가진 데이터를 모든 프로세스에 뿌려줍니다 (Broadcast).
                accelerator.wait_for_everyone()
                torch.distributed.broadcast(batch, src=0)
                accelerator.wait_for_everyone()
                
                # 모델 연산은 모든 프로세스가 함께 수행합니다.
                output = model(batch,
                            use_cache=False,
                            output_hidden_states=False,
                            output_attentions=False)[0]
                
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:]
                
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.to(shift_logits.device).view(-1))
                
                # loss 값은 각자 계산되지만, 집계는 메인 프로세스에서만 수행합니다.
                if accelerator.is_main_process:
                    acc_loss += loss.item()
                    progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")
        
        # 모든 계산이 끝날 때까지 기다립니다.
        accelerator.wait_for_everyone()

        # 최종 결과 계산 및 저장은 메인 프로세스에서만 수행합니다.
        if accelerator.is_main_process:
            avg_loss = acc_loss / nsamples
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            accelerator.print(f'{dataset} perplexity: {ppl:.3f}')
            
            result_path = ptq_args.save_qmodel_path if ptq_args.save_qmodel_path not in [None, ''] else ptq_args.load_qmodel_path
            result_path = result_path.split('.pth')[0] + f'_ppl_results.json'
            
            try:
                with open(result_path, 'r') as f:
                    comp_result= json.load(f)
            except:
                comp_result = {}
                comp_result['ppl'] = {}
            if 'ppl' not in comp_result or not isinstance(comp_result['ppl'], dict):
                comp_result['ppl'] = {}
            comp_result['ppl'][dataset] = ppl
            
            with open(result_path, 'w') as f:
                json.dump(comp_result, f, indent=4)

if __name__ == "__main__":
    train()