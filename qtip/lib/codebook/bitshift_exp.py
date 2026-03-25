import itertools
import math
import os
from functools import cache

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from lib.codebook import kdict
from lib.utils.kernel_check import has_kernel
from lib.utils.kernel_decompress import decode_compressed
from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

from scipy.cluster.vq import kmeans2, ClusterError
import numpy as np

def _valid_tlut(tlut, k, V):
    return isinstance(tlut, torch.Tensor) and tlut.ndim == 2 and tlut.shape == (k, V)

def _build_tlut_kmeans2(k, V, fname, _sample_like, scale_after=True, max_tries=50, iters=20):
    # 1) 캐시 먼저 로드(깨졌으면 폐기)
    if os.path.exists(fname):
        try:
            tlut = torch.load(fname)
            if _valid_tlut(tlut, k, V):
                return tlut
        except Exception:
            pass
        # 깨진 캐시(예: 8191개)면 삭제 권장
        try:
            os.remove(fname)
        except Exception:
            pass

    # 2) 데이터는 한 번만 만들고, init만 재시도
    data = _sample_like((1 << 20, V))  # gaussian/laplace에 맞게 샘플링
    data_np = np.asarray(data.cpu().numpy(), dtype=np.float32, order="C")

    last_err = None
    for _ in range(max_tries):
        init = _sample_like((k, V))
        init_np = np.asarray(init.cpu().numpy(), dtype=np.float32, order="C")
        try:
            centroids, labels = kmeans2(
                data_np,
                init_np,
                iter=iters,
                minit="matrix",     # init_np 그대로 초기 centroid로 사용
                missing="raise",    # 빈 클러스터 발생 시 즉시 예외
                check_finite=True,
            )
        except ClusterError as e:
            last_err = e
            continue

        if centroids.shape[0] != k:
            # 혹시 모를 비정상 출력 방어
            last_err = RuntimeError(f"centroids.shape[0]={centroids.shape[0]} != k={k}")
            continue

        tlut = torch.from_numpy(centroids).to(torch.float32)
        if scale_after:
            tlut = (tlut / tlut.std(unbiased=False)) * 0.9682458365518543
        torch.save(tlut, fname)
        return tlut

    raise RuntimeError(f"kmeans2 failed after {max_tries} tries. last_err={last_err}")

def decode_1mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_2mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 264435761 + 1013904223
    x = x & ((1 << 32) - 1)
    x = ((x * 1664525) >> 32) + x
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_3inst(x):

    def bfe16_to_fp16(x):
        x[torch.where(x >= 2**15)] -= 2**16
        return torch.tensor(x.to(torch.int16).numpy().view(np.float16))

    a = 89226354
    b = 64248484
    fpmask = 996162400
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bottom = bfe16_to_fp16(res & ((1 << 16) - 1))
    return (top + bottom).float()


def quantlut(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L)
        lut = (lut + 1) * lut
        lut = (lut >> (16 - nbits)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    return lut


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


class bitshift_codebook(nn.Module):
    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 tlut_bits=16,
                 decode_mode='lut',
                 tlut=None,
                 init_dist: str = "gaussian",   # "gaussian" | "laplace"
                 laplace_scale: float | None = None):
        super(bitshift_codebook, self).__init__()
        self.idx_dtype = torch.int32
        self.opt_scale = 1

        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode

        self.init_dist = init_dist
        # 기본값: 가우시안 std=1과 대략 스케일 맞추기(라플라스 분산=2*b^2)
        self.laplace_scale = float(laplace_scale) if laplace_scale is not None else (1.0 / math.sqrt(2.0))

        def _sample_like(shape, dtype=torch.float32, device=None):
            if self.init_dist == "gaussian":
                return torch.randn(*shape, dtype=dtype, device=device)
            elif self.init_dist == "laplace":
                loc = torch.tensor(0.0, dtype=dtype, device=device)
                scale = torch.tensor(self.laplace_scale, dtype=dtype, device=device)
                return torch.distributions.Laplace(loc, scale).sample(shape)
            else:
                raise ValueError(f"Unknown init_dist={self.init_dist} (use 'gaussian' or 'laplace')")

        if decode_mode == 'lut':
            if tlut is None:
                assert tlut_bits == L
                # BEFORE: torch.randn(2**L, V)
                self.register_buffer('tlut', _sample_like((2**L, V)))
                self.register_buffer('lut', self.tlut.T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        elif decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_1mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '2mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_2mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '3inst':
            assert V == 1
            self.register_buffer('lut',
                                 decode_3inst(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == 'quantlut':
            if tlut is None:
                assert tlut_bits > 0
                if V == 1:
                    tlut = torch.erfinv((torch.arange(1 << tlut_bits) + 0.5) /
                                        (1 << tlut_bits) * 2 -
                                        1) * torch.tensor(2.0).sqrt()
                    # tlut = tlut.unsqueeze(-1) ## ryu
                elif V == 2:
                    n = 2**tlut_bits
                    tlut = torch.zeros(n)
                    R = ((n / (n - torch.arange(n))).log() * 2).sqrt()
                    tlut = torch.stack(
                        [R * torch.arange(n).sin(), R * torch.arange(n).cos()],
                        dim=-1)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut.unsqueeze(-1)) ## ryu
                # self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()

        elif decode_mode == 'quantlut_sym':
            if tlut is None:
                assert tlut_bits > 0

                # 분포/스케일별로 캐시 파일 분리(기존 gaussian 캐시와 충돌 방지)
                dist_tag = self.init_dist
                if self.init_dist == "laplace":
                    s = f"{self.laplace_scale}".replace(".", "p")
                    dist_tag = f"laplace_s{s}"
                fname = f'/tmp/kmeans_{tlut_bits}_{V}_{dist_tag}.pt'

                if V == 2:
                    if not os.path.exists(fname):
                    # if True:
                        # BEFORE: tlut = torch.randn(2**tlut_bits, V)
                        tlut0 = _sample_like((2**tlut_bits, V))
                        import scipy
                        # BEFORE: data = torch.randn(1 << 20, 2)
                        data = _sample_like((1 << 20, 2))
                        clusters = scipy.cluster.vq.kmeans2(data.numpy(), tlut0.numpy())
                        tlut = torch.tensor(clusters[0])
                        tlut = (tlut / tlut.std(unbiased=False)) * 0.9682458365518543
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                    # k = 1 << tlut_bits
                    # dist_tag = self.init_dist
                    # if self.init_dist == "laplace":
                    #     s = f"{self.laplace_scale}".replace(".", "p")
                    #     dist_tag = f"laplace_s{s}"
                    # fname = f"/tmp/kmeans_{tlut_bits}_{V}_{dist_tag}.pt"

                    # tlut = _build_tlut_kmeans2(
                    #     k=k, V=2, fname=fname, _sample_like=_sample_like,
                    #     scale_after=True, max_tries=50, iters=20
                    # )
                    # assert tlut.shape[0] == k
                elif V == 1:  # ryu
                    if not os.path.exists(fname):
                        # BEFORE: tlut = torch.randn(2**tlut_bits, 1)
                        tlut0 = _sample_like((2**tlut_bits, 1))
                        import scipy
                        # BEFORE: data = torch.randn(1 << 20, 1)
                        data = _sample_like((1 << 20, 1))
                        clusters = scipy.cluster.vq.kmeans(data.numpy(), tlut0.numpy())
                        tlut = torch.tensor(clusters[0])
                        tlut = (tlut / tlut.std(unbiased=False))
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                else:
                    raise Exception

                self.register_buffer('tlut', tlut)
                self.register_buffer('lut', quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        else:
            raise Exception

        self.fakeinf = torch.tensor(torch.inf)

        self.register_buffer('sumdelta',
                             torch.arange(2**(K * V)) << (L - K * V))
        self.sumdelta = self.sumdelta.view(1, 1, -1)

        self.register_buffer('state', torch.arange(2**L).unsqueeze(0))
        self.register_buffer('state_cand',
                             (self.state >>
                              (K * V))[0, ::2**(K * V)].unsqueeze(-1) +
                             self.sumdelta)
        self.register_buffer('recons_state', self.recons(self.state))

        self.version = 0

    def recons_lut(self):
        if self.decode_mode == 'lut':
            self.lut = self.tlut.T.contiguous()
        elif self.decode_mode == 'quantlut':
            self.lut = quantlut(self.tlut, self.L,
                                self.tlut_bits).T.contiguous()
        elif self.decode_mode == 'quantlut_sym':
            self.lut = quantlut_sym(self.tlut, self.L,
                                    self.tlut_bits).T.contiguous()

    def recons(self, encoded, **kwargs):
        return self.lut[:,
                        encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, thing):
        state_err = (self.recons_state -
                     thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2**(self.K * self.V)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2**(self.K * self.V)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1,
            best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.V == 0
        # cost is (B, 2**L)
        cost = (self.recons_state -
                X[:self.V].unsqueeze(-1)).square().sum(dim=0)

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap <<
                     (self.K * self.V)).unsqueeze(-1) + torch.arange(
                         2**(self.K * self.V)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        from_state = torch.zeros(T // self.V,
                                 B,
                                 2**(self.L - self.K * self.V),
                                 dtype=self.state.dtype,
                                 device=self.state.device)

        for i in range(1, T // self.V):
            from_state[i], cost = self.update(cost,
                                              X[i * self.V:(i + 1) * self.V])

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        final_state = torch.zeros(T // self.V,
                                  B,
                                  dtype=self.idx_dtype,
                                  device=X.device)
        final_state[T // self.V - 1] = torch.argmin(cost, dim=-1)
        for i in range(T // self.V - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1,
                (final_state[i].to(torch.int64).unsqueeze(-1)) >>
                (self.K * self.V))[..., 0]
        return final_state

    def quantize_seq(self, X, overlap=None, **kwargs):
        T, NO = X.shape
        bs = min(2**(24 - self.L), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)

        Qidxs = torch.zeros(N // bs,
                            T // self.V,
                            bs,
                            dtype=self.idx_dtype,
                            device=X.device)
        for i in range(len(X)):
            b_overlap = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=b_overlap)
        Qidxs = Qidxs.transpose(0, 1).reshape(T // self.V, N)[:, :NO]
        return Qidxs

    def quantize(self, X, **kwargs):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.V) * self.V, 0)
        state = self.quantize_seq(roll_X, overlap=None)
        overlap = state[T // (2 * self.V)] >> self.K * self.V
        state = self.quantize_seq(X, overlap=overlap)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(
            X.device)

    def pack_trellis(self, trellis):
        # T is really T // self.V here
        B, T = trellis.shape
        bf = torch.zeros(B,
                         T * self.K * self.V + self.L - self.K * self.V,
                         dtype=bool,
                         device=trellis.device)
        bf[:, :self.L] = (trellis[:, 0].unsqueeze(-1) & (2**torch.arange(
            self.L, device=trellis.device).flip(dims=(-1, ))).unsqueeze(0)) > 0
        K_mask = 2**torch.arange(
            self.K * self.V,
            device=trellis.device).flip(dims=(-1, )).unsqueeze(0)
        for i in range(1, T):
            assert ((trellis[:, i - 1] &
                     ((1 << (self.L - self.K * self.V)) - 1)) == (
                         trellis[:, i] >> (self.K * self.V))).all()
            bf[:,
               (self.L +
                (i - 1) * self.K * self.V):(self.L + i * self.K * self.V)] = (
                    (trellis[:, i] &
                     ((1 <<
                       (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask) > 0

        bf = bf[:, :-(self.L - self.K * self.V)]
        pad_amt = math.ceil(
            T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(
            -1, (T * self.K * self.V + pad_amt) // 16, 16)

        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=bf.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf_sum = (bf.to(torch.int32) * uint_mask).sum(dim=-1)
        return bf_sum.to(torch.uint16)

    def unpack_trellis(self, packed, T):
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=packed.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(T * self.K / 16) * 16 - T * self.K
        bf = bf.reshape(-1, (T * self.K + pad_amt))[:, :T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2**torch.arange(
            self.L, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        K_mask = (2**torch.arange(
            self.K * self.V, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0],
                              T // self.V,
                              dtype=torch.int32,
                              device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, T // self.V):
            trellis[:, i] = ((trellis[:, i-1] << (self.K*self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i-1)*self.K*self.V : self.L + i*self.K*self.V].int() * K_mask).sum(dim=-1)

        return trellis


class BitshiftLinear(nn.Module):

    def __init__(self,
                 td_x,
                 td_y,
                 L,
                 K,
                 V,
                 tlut_bits,
                 decode_mode,
                 dtype=torch.float16,
                 tlut=None,
                 has_kernel=False):
        super().__init__()
        self.td_x = td_x
        self.td_y = td_y
        self.V = V
        self.cb = bitshift_codebook(L, K, V, tlut_bits, decode_mode, tlut=tlut)
        self.internal_dtype = dtype
        self.has_kernel = has_kernel
        self.scale = 32

    def get_hatW(self, unpacked_trellis, m, n):
        return self.cb.recons(unpacked_trellis).transpose(0, 1).transpose(
            1, 2).reshape(m // self.td_x, n // self.td_y, self.td_x,
                          self.td_y).transpose(1, 2).reshape(m, n)

    # def get_hatW_kernel(self, trellis, m, n):
    #     out = decode_compressed(self.cb.L, self.cb.tlut_bits, self.cb.K,
    #                             int(math.log2(self.V)), m, n, trellis.view(-1),
    #                             self.cb.lut.T)
    #     return out

    def get_hatW_kernel(self, trellis, m, n):
            """
            decode_compressed 실행 시간을 측정하여 결과와 함께 반환합니다.
            """
            timings = {}
            
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            # 추후 decode_compressed 내부를 수정하면, 여기서 (out, inner_timings)를 받아야 할 수도 있습니다.
            out = decode_compressed(self.cb.L, self.cb.tlut_bits, self.cb.K,
                                    int(math.log2(self.V)), m, n, trellis.view(-1),
                                    self.cb.lut.T)
            end_evt.record()
            torch.cuda.synchronize()
            
            timings['decode_compressed_kernel_call'] = start_evt.elapsed_time(end_evt)
            
            return out, timings

    def cache_hatW(self, packed_trellis, had_left, had_right, K_left, K_right,
                   m, n, rcp, tp_rank):
        """
        각 단계별(Decompression, Scaling, Hadamard Transform) 시간을 측정하고
        self.hatW를 설정한 뒤, timing dictionary를 반환합니다.
        """
        total_timings = {}
        
        # 이벤트 객체 생성
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)

        # 1. Kernel Decompression
        if self.has_kernel:
            hatW, decomp_timings = self.get_hatW_kernel(packed_trellis, m, n)
            total_timings.update(decomp_timings)
        else:
            raise
            t_start.record()
            hatW = self.get_hatW(
                self.cb.unpack_trellis(packed_trellis, self.td_x * self.td_y),
                m, n)
            t_end.record()
            torch.cuda.synchronize()
            total_timings['get_hatW_cpu'] = t_start.elapsed_time(t_end)

        # 2. Casting & Scaling (Float conversion)
        t_start.record()
        hatW = hatW.float() / self.scale
        t_end.record()
        torch.cuda.synchronize()
        total_timings['scaling_and_cast_float'] = t_start.elapsed_time(t_end)

        # 3. Hadamard Transform & Final Casting
        t_start.record()
        if rcp == 1:
            raise  # 주석 처리된 부분이라도 흐름상 에러가 날 수 있어 원본 유지 또는 해제 필요
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW.reshape(tp_rank * m, n // tp_rank),
                                 had_left, K_left).reshape(m, n).T, had_right,
                K_right).T.contiguous().to(self.internal_dtype)
        elif rcp == 2:
            raise
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW, had_left,
                                 K_left).T.reshape(tp_rank * n,
                                                   m // tp_rank), had_right,
                K_right).reshape(n, m).T.contiguous().to(self.internal_dtype)
        else:
            self.hatW = matmul_hadU_cuda(
                matmul_hadU_cuda(hatW, had_left, K_left).T, had_right,
                K_right).T.contiguous().to(self.internal_dtype)
        
        t_end.record()
        torch.cuda.synchronize()
        total_timings['hadamard_and_final_cast'] = t_start.elapsed_time(t_end)

        return total_timings

    # def cache_hatW(self, packed_trellis, had_left, had_right, K_left, K_right,
    #                m, n, rcp, tp_rank):
    #     if self.has_kernel:
    #         hatW = self.get_hatW_kernel(packed_trellis, m, n)
    #     else:
    #         raise
    #         # hatW = self.get_hatW(
    #         #     self.cb.unpack_trellis(packed_trellis, self.td_x * self.td_y),
    #         #     m, n)
    #     hatW = hatW.float() / self.scale

    #     if rcp == 1:
    #         raise
    #         self.hatW = matmul_hadU_cuda(
    #             matmul_hadU_cuda(hatW.reshape(tp_rank * m, n // tp_rank),
    #                              had_left, K_left).reshape(m, n).T, had_right,
    #             K_right).T.contiguous().to(self.internal_dtype)
    #     elif rcp == 2:
    #         raise
    #         self.hatW = matmul_hadU_cuda(
    #             matmul_hadU_cuda(hatW, had_left,
    #                              K_left).T.reshape(tp_rank * n,
    #                                                m // tp_rank), had_right,
    #             K_right).reshape(n, m).T.contiguous().to(self.internal_dtype)
    #     else:
    #         self.hatW = matmul_hadU_cuda(
    #             matmul_hadU_cuda(hatW, had_left, K_left).T, had_right,
    #             K_right).T.contiguous().to(self.internal_dtype)

    def forward(self,
                input,
                trellis,
                SU,
                SV,
                had_left,
                had_right,
                K_left,
                K_right,
                rcp,
                tp_rank,
                mode='eval',
                **kwargs):
        n, m = len(SU), len(SV)
        x = input.view(-1, n).to(torch.float32)
        x = x * SU

        if mode == 'train-fixW':
            x = (x.to(self.internal_dtype) @ self.hatW.T).float()
        else:
            bs = x.shape[0]

            if rcp == 1:
                x = matmul_hadUt_cuda(x.reshape(-1, n // tp_rank), had_left,
                                      K_left).reshape(x.shape) / self.scale
            else:
                x = matmul_hadUt_cuda(x, had_left, K_left) / self.scale

            if bs == 1 and self.has_kernel:
                wrapper = getattr(
                    torch.ops.quip_lib,
                    f"decompress_matvec_qtip_{m}_1_{x.numel()}_{self.cb.K}")

                x = wrapper(trellis, x, self.cb.tlut)

            else:
                if mode == 'train-recons':
                    self.cb.recons_lut()

                if self.has_kernel:
                    x = BitshiftLinearKernelAG.apply(
                        x, trellis, m, n, self.cb.L, self.cb.tlut_bits, self.cb.K,
                        self.V, self.cb.lut).float()
                else:
                    if mode == 'eval':
                        trellis = self.cb.unpack_trellis(
                            trellis, self.td_x * self.td_y)
                    hatW = self.get_hatW(trellis, m, n)
                    x = (x.to(hatW.dtype) @ hatW.T).float()

            if rcp == 2:
                x = matmul_hadU_cuda(x.reshape(-1, m // tp_rank), had_right,
                                     K_right).reshape(x.shape)
            else:
                x = matmul_hadU_cuda(x, had_right, K_right)

        x = x.to(SV.device) * (SV * self.scale)
        return x.view(*input.shape[:-1], m).to(input.dtype)


        
class BitshiftLinearKernelAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, trellis, m, n, L, tlut_bits, K, V, lut):
        ctx.save_for_backward(trellis, lut)
        ctx.L = L
        ctx.tlut_bits = tlut_bits
        ctx.K = K
        ctx.V = V
        ctx.m = m
        ctx.n = n
        
        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)),
                                 m, n, trellis.view(-1), lut.T)
        return input.to(hatW.dtype) @ hatW.T

    @staticmethod
    def backward(ctx, grad_output):
        trellis, lut = ctx.saved_tensors
        L = ctx.L
        tlut_bits = ctx.tlut_bits
        K = ctx.K
        V = ctx.V
        m = ctx.m
        n = ctx.n

        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)),
                                 m, n, trellis.view(-1), lut.T)
        
        grad_input = grad_output.to(hatW.dtype) @ hatW
        return grad_input, None, None, None, None, None, None, None, None
