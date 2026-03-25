import torch

def get_had180():
    """
    180차원 Hadamard 행렬 생성 (Correct Paley Construction Type I)
    N = 180 = 179 + 1 (179 is prime, 179 ≡ 3 mod 4)
    
    구성 방식: H = S + I
    (S는 Skew-Hadamard Matrix)
    """
    p = 179
    
    # 1. Legendre Symbol (이차 잉여) 계산
    # legendre[i] = 1 (이차잉여), -1 (비이차잉여), 0 (0인 경우)
    legendre = [0] * p
    for i in range(1, p):
        # 오일러 판정법: a^((p-1)/2) mod p
        val = pow(i, (p - 1) // 2, p)
        if val == p - 1: # mod p 에서 -1
            legendre[i] = -1
        else:
            legendre[i] = val

    # 2. Jacobsthal Matrix Q 생성
    # Q_ij = legendre((i - j) % p)
    # p=179 ≡ 3 (mod 4) 이므로 Q는 Skew-symmetric (Q^T = -Q)
    indices = torch.arange(p).view(-1, 1) - torch.arange(p).view(1, -1)
    indices = indices % p
    
    leg_tensor = torch.tensor(legendre, dtype=torch.float32)
    Q = leg_tensor[indices]

    # 3. Skew-Hadamard Matrix S 구성
    # S = | 0   1^T |
    #     | -1   Q  |
    S = torch.zeros((p + 1, p + 1), dtype=torch.float32)
    
    # 첫 행은 1 (S_00은 0)
    S[0, 1:] = 1.0
    # 첫 열은 -1 (S_00은 0)
    S[1:, 0] = -1.0
    # 나머지 블록은 Q
    S[1:, 1:] = Q

    # 4. 최종 Hadamard Matrix H = S + I
    # (Skew-Hadamard 행렬에 단위행렬을 더하면 Hadamard 행렬이 됨)
    I = torch.eye(p + 1, dtype=torch.float32)
    H = S + I

    return H

# --- 검증용 코드 (실행 시 주석 해제하여 확인 가능) ---
if __name__ == "__main__":
    # h = get_had180()
    # # H @ H.T 계산
    # gram = h @ h.T
    
    # # 대각선은 180, 나머지는 0이어야 함
    # is_orthogonal = torch.allclose(gram, 180 * torch.eye(180), atol=1e-4)
    
    # print(f"Shape: {h.shape}") # [180, 180]
    # print(f"Is Orthogonal?: {is_orthogonal}") # True여야 함
    
    # # 실패했던 부분 확인 (첫 행과 두번째 행 내적)
    # print(f"Row 0 dot Row 1: {torch.dot(h[0], h[1])}") # 0.0 이어야 함
    
    # print(torch.matmul(h, h.T)) # 대각선이 180인 행렬이 나와야 함
    # print(h)

    H = get_had180()

    # 1) 크기 확인
    assert H.shape == (180, 180)

    # 2) 원소가 ±1만인지 확인
    assert torch.all((H == 1) | (H == -1)), (H.min().item(), H.max().item(), torch.unique(H))

    # 3) 직교성 확인: H H^T = 180 I
    G = H @ H.T
    I = 180 * torch.eye(180, dtype=H.dtype)
    assert torch.equal(G, I), (G - I).abs().max().item()

    print("OK: Hadamard(180) verified.")