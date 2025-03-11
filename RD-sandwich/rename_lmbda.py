import os
import re

# 대상 디렉토리 설정
target_dir = "/home/jgryu/Weight_compression/RD-sandwich/checkpoints_v3/llama3_8B_per_tensor/model-layers-0-self_attn-v_proj-weight"

# 정규식 패턴 정의 (lmbda=<숫자>)
pattern = re.compile(r"lmbda=([0-9]+)")

# 하위 디렉토리를 재귀적으로 탐색
for root, dirs, files in os.walk(target_dir, topdown=False):  # 하위 디렉토리부터 처리
    for dir_name in dirs:
        old_path = os.path.join(root, dir_name)

        # lmbda=<integer> 형식을 찾음
        match = pattern.search(dir_name)
        if match:
            # 새로운 디렉토리 이름 생성
            new_dir_name = pattern.sub(r"lmbda=\1.0", dir_name)
            new_path = os.path.join(root, new_dir_name)

            # 이름 변경
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except OSError as e:
                print(f"Failed to rename {old_path}: {e}")
