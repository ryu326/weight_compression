# from huggingface_hub import snapshot_download

# repo_id = "meta-llama/Llama-2-70b-hf"
# local_dir = "../Wparam_dataset/hf_model/meta-llama--Llama-2-70b-hf_"

# # 전체 모델 다운로드를 검증하고 빠진 파일만 이어받기
# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     resume_download=True,  # 이어받기 기능 활성화
#     local_dir_use_symlinks=False
# )

# print(f"'{repo_id}' 모델 전체 다운로드 검증 및 완료!")


from huggingface_hub import hf_hub_download

# 다운로드할 파일 정보
repo_id = "meta-llama/Llama-2-70b-hf"
filename = "model-00006-of-00015.safetensors"

# 파일을 저장할 로컬 경로 (에러 메시지에 나온 경로와 일치시켜야 합니다)
local_dir = "../Wparam_dataset/hf_model/meta-llama--Llama-2-70b-hf_"

# 특정 파일 다운로드 실행
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    # local_dir_use_symlinks=False # 필요한 경우 심볼릭 링크 사용 방지
)

print(f"'{filename}' 다운로드 완료! 저장 위치: {local_dir}")