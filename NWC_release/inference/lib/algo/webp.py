import os
import subprocess
import sys
import torch
import numpy as np
from PIL import Image
import re

# cwebp/dwebp 실행 파일 경로를 환경 변수에서 가져오거나 기본값을 사용
CWEBP = os.environ.get("CWEBP", "cwebp")
DWEBP = os.environ.get("DWEBP", "dwebp")

def check_webp_availability():
    """cwebp와 dwebp 프로그램이 사용 가능한지 확인하는 함수"""
    for cmd, name in [(CWEBP, "cwebp"), (DWEBP, "dwebp")]:
        try:
            subprocess.call([cmd, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print(f"*** 오류: WebP 프로그램 '{name}'을(를) 찾을 수 없습니다.")
            print(f"'{name}'을(를) 설치하고 시스템의 PATH에 추가하거나, 환경 변수를 설정해주세요.")
            print(" (Debian/Ubuntu: sudo apt-get install webp)")
            sys.exit(1)

def compress_tensor_with_webp(tensor_in: torch.Tensor, q: int, tmp_dir: str = ".tmp_webp"):
    """
    2D 텐서를 WebP로 압축하고 다시 텐서로 복원합니다.

    Args:
        tensor_in (torch.Tensor): 입력 2D 텐서 (uint8 타입이어야 함)
        q (int): WebP 품질 파라미터 (0-100, 높을수록 고품질)
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
    path_webp = os.path.join(tmp_dir, f"{base_name}_q{q}.webp")
    path_out_png = os.path.join(tmp_dir, f"{base_name}_decoded.png")

    try:
        # 1. 텐서를 PNG 파일로 저장
        Image.fromarray(tensor_in.cpu().numpy()).save(path_in_png)

        # 2. PNG를 WebP로 압축 및 전체 비트 수 계산
        cmd = [CWEBP, "-q", str(q), path_in_png, "-o", path_webp]
        # cwebp는 통계를 stderr로 출력하는 경우가 많음
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"cwebp failed:\n{result.stderr}")
        
        # cwebp 출력에서 bpp 정보 파싱하여 총 비트 수 계산
        total_bits = _parse_webp_output_for_bits(result.stderr, tensor_in.shape)

        # 3. WebP를 PNG로 디코딩
        subprocess.run([DWEBP, path_webp, "-o", path_out_png], check=True)

        # 4. 디코딩된 PNG를 텐서로 복원
        img_out = Image.open(path_out_png).convert('L')
        tensor_out = torch.from_numpy(np.array(img_out))

        return tensor_out, total_bits

    finally:
        # 임시 파일 정리
        for f in [path_in_png, path_webp, path_out_png]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)


# ---------------- WebP 출력 파싱을 위한 헬퍼 함수들 ----------------

def _match_regex_ungroup_as_int(s, r):
    """정규식으로 숫자를 찾아 int로 반환하는 헬퍼"""
    m = re.search(r, s)
    if not m:
        # cwebp는 -quiet 옵션이 없으면 항상 통계를 출력하므로, 출력이 없다면 오류일 수 있음
        raise ValueError(f"정규식 '{r}'이(가) cwebp 출력에서 일치하는 항목을 찾지 못했습니다:\n{s}")
    otp = tuple(map(int, m.groups()))
    return otp[0] if len(otp) == 1 else otp


def _parse_webp_output_for_bits(cwebp_stderr: str, shape: torch.Size) -> int:
    """cwebp의 출력(stderr)을 파싱하여 총 비트 수를 계산합니다."""
    # cwebp 출력 예시:
    # Dimension: 256 x 256
    # Output:    564 bytes Y-U-V-All-PSNR 43.85 49.33 49.56   44.92 dB
    # ...
    # bytes used:  header:             24  (4.3%)
    # try:
    total_bytes = _match_regex_ungroup_as_int(cwebp_stderr, r"Output:\s+(\d+)\s+bytes")
    header_bytes = _match_regex_ungroup_as_int(cwebp_stderr, r"header:\s+(\d+)")
    content_bytes = total_bytes - header_bytes
    return content_bytes * 8
    # except ValueError:
    #     # 통계 출력이 없는 경우(예: 이미지가 너무 작거나 q=100일 때) 파일 크기를 직접 사용
    #     h, w = shape
    #     bpp = _match_regex_ungroup_as_int(cwebp_stderr, r"(\d+\.\d+)\s+bits/pixel")[0]
    #     return int(bpp * w * h)


# ---------------- 실행 예제 ----------------

if __name__ == '__main__':
    # 0. cwebp, dwebp 프로그램이 설치되어 있는지 확인
    check_webp_availability()

    # 1. 압축할 2D 텐서 생성 (uint8 타입)
    original_tensor = (torch.rand(256, 256) * 255).to(torch.uint8)
    
    # 2. WebP 압축 품질 설정
    quality_q = 100  # 0-100 사이, 높을수록 고품질

    print(f"원본 텐서: shape={original_tensor.shape}, dtype={original_tensor.dtype}")
    print(f"WebP 압축 시작 (q={quality_q})...")

    # 3. WebP 압축 및 복원 함수 실행
    reconstructed_tensor, bits_used = compress_tensor_with_webp(original_tensor, quality_q)

    print("\n--- 결과 ---")
    print(f"복원된 텐서: shape={reconstructed_tensor.shape}, dtype={reconstructed_tensor.dtype}")
    print(f"압축에 사용된 총 비트 수: {bits_used} bits")
    
    # 4. 원본과 복원된 텐서 간의 오차 계산 (Mean Absolute Error)
    mae = torch.mean(torch.abs(original_tensor.float() - reconstructed_tensor.float()))
    print(f"평균 절대 오차 (MAE): {mae.item():.4f}")