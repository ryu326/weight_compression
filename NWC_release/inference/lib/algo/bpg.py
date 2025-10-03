import os
import subprocess
import sys
import torch
import numpy as np
from PIL import Image

# bpgenc/bpgdec 실행 파일 경로를 환경 변수에서 가져오거나 기본값을 사용
BPGENC = os.environ.get("BPGENC", "bpgenc")
BPGDEC = os.environ.get("BPGDEC", "bpgdec")

def check_bpg_availability():
    """bpgenc와 bpgdec 프로그램이 사용 가능한지 확인하는 함수"""
    for cmd, name in [(BPGENC, "bpgenc"), (BPGDEC, "bpgdec")]:
        try:
            subprocess.call([cmd, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print(f"*** 오류: BPG 프로그램 '{name}'을(를) 찾을 수 없습니다.")
            print(f"{name}을(를) 설치하고 시스템의 PATH에 추가하거나, 환경 변수를 설정해주세요.")
            sys.exit(1)

def compress_tensor_with_bpg(tensor_in: torch.Tensor, q: int, tmp_dir: str = ".tmp_bpg"):
    """
    2D 텐서를 BPG로 압축하고 다시 텐서로 복원합니다.

    Args:
        tensor_in (torch.Tensor): 입력 2D 텐서 (uint8 타입이어야 함)
        q (int): BPG 양자화 파라미터 (0-51, 낮을수록 고품질)
        tmp_dir (str): 임시 파일을 저장할 디렉토리

    Returns:
        (torch.Tensor, int): (복원된 텐서, 압축에 사용된 총 비트 수)
    """
    if tensor_in.dtype != torch.uint8 or tensor_in.dim() != 2:
        raise ValueError("입력 텐서는 2차원의 uint8 타입이어야 합니다.")

    os.makedirs(tmp_dir, exist_ok=True)
    
    # 임시 파일 경로 정의
    base_name = "temp_tensor"
    path_in_png = os.path.join(tmp_dir, f"{base_name}.png")
    path_bpg = os.path.join(tmp_dir, f"{base_name}_q{q}.bpg")
    path_out_png = os.path.join(tmp_dir, f"{base_name}_decoded.png")

    try:
        # 1. 텐서를 PNG 파일로 저장
        Image.fromarray(tensor_in.cpu().numpy()).save(path_in_png)

        # 2. PNG를 BPG로 압축
        subprocess.call([BPGENC, "-q", str(q), path_in_png, "-o", path_bpg])

        # 2.1. 전체 비트 수 계산
        info = bpg_image_info(path_bpg)
        total_bits = info.num_bytes_for_picture * 8

        # 3. BPG를 PNG로 디코딩
        # subprocess.call([BPGDEC, "-o", path_out_png, path_bpg])
        # 수정 후:
        result = subprocess.run([BPGDEC, "-o", path_out_png, path_bpg], 
                                capture_output=True, text=True)
        
        # 만약 디코딩 과정에서 오류가 발생했다면
        if result.returncode != 0:
            print("--- BPG 디코딩 오류 발생 ---")
            print("Return Code:", result.returncode)
            print("Stderr:", result.stderr) # 오류 메시지 출력
            print("Stdout:", result.stdout)
            # 오류 발생 시 여기서 중단하거나 예외를 발생시킬 수 있습니다.
            raise RuntimeError("bpgdec failed")

        # 4. 디코딩된 PNG를 텐서로 복원
        img_out = Image.open(path_out_png)
        tensor_out = torch.from_numpy(np.array(img_out))

        return tensor_out, total_bits

    finally:
        # 임시 파일 정리
        for f in [path_in_png, path_bpg, path_out_png]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)


# ---------------- BPG 파일 파싱을 위한 헬퍼 함수들 ----------------

class BPGImageInfo:
    def __init__(self, width, height, num_bytes_for_picture):
        self.width = width
        self.height = height
        self.num_bytes_for_picture = num_bytes_for_picture

def bpg_image_info(p: str) -> BPGImageInfo:
    with open(p, "rb") as f:
        magic = f.read(4)
        expected_magic = bytearray.fromhex("425047fb")
        assert magic == expected_magic, f"Not a BPG file: {p}"
        f.read(2)
        width = _read_ue7(f)
        height = _read_ue7(f)
        picture_data_length = _read_ue7(f)
        
        current_pos = f.tell()
        f.seek(0, 2)
        end_pos = f.tell()
        f.seek(current_pos)
        
        num_bytes_for_picture = (end_pos - current_pos) if picture_data_length == 0 else picture_data_length
        return BPGImageInfo(width, height, num_bytes_for_picture)

def _read_ue7(f):
    bits = 0
    while True:
        byte_val = f.read(1)[0]
        bits = (bits << 7) | (byte_val & 0x7F)
        if (byte_val & 0x80) == 0:
            return bits

# ---------------- 실행 예제 ----------------

if __name__ == '__main__':
    # 0. bpgenc, bpgdec 프로그램이 설치되어 있는지 확인
    check_bpg_availability()

    # 1. 압축할 2D 텐서 생성 (uint8 타입)
    original_tensor = (torch.rand(256, 256) * 255).to(torch.uint8)
    
    # 2. BPG 압축 품질 설정
    quality_q = 28

    print(f"원본 텐서: shape={original_tensor.shape}, dtype={original_tensor.dtype}")
    print(f"BPG 압축 시작 (q={quality_q})...")

    # 3. BPG 압축 및 복원 함수 실행
    reconstructed_tensor, bits_used = compress_tensor_with_bpg(original_tensor, quality_q)

    print("\n--- 결과 ---")
    print(f"복원된 텐서: shape={reconstructed_tensor.shape}, dtype={reconstructed_tensor.dtype}")
    print(f"압축에 사용된 총 비트 수: {bits_used} bits")
    
    # 4. 원본과 복원된 텐서 간의 오차 계산 (Mean Absolute Error)
    mae = torch.mean(torch.abs(original_tensor.float() - reconstructed_tensor.float()))
    print(f"평균 절대 오차 (MAE): {mae.item():.4f}")