from .vqvae import VQVAE
from .vqvae_idx import VQVAE_IDX
from .vqvae_calib_mag import VQVAE_MAG
from .vqvae_scale import VQVAE_SCALE
from .vqvae_idx_mag import VQVAE_IDX_MAG
from .nwc import SimpleVAECompressionModel, MeanScaleHyperprior, SimpleVAECompressionModel_with_Transformer
from .nwc_ql import NWC_ql, NWC_ql_LTC
from .nwc_ql_cdt import NWC_ql_conditional
from .nwc_ql_v2 import NWC_ql_learnable_scale
from .nwc_ql_sga import NWC_ql_SGA, NWC_ql_SGA_Vbr
from .nwc_qmap import NWC_qmap, NWC_qmap2, NWC_qmap3
from .nwc_scale_cond import NWC_scale_cond, NWC_scale_cond_ltc
from .nwc_scale_cond_v2 import NWCScaleCond
from .nwc_lora import LoRACompressionModel
# from .nwc_ql_cdt import NWC_conditional, NWC_conditional2
# from .nwc_ql_cdt_ln import NWC_conditional_ln
# from .nwc_hess import SimpleVAECompressionModel_hess
# from .nwc_ql_batchnorm import NWC_ql_bn
# from .nwc_bn import SimpleVAECompressionModel_bn

def get_model(model_class, opts, scale, shift):
    if not hasattr(opts, 'use_hyper'):
        opts.use_hyper = False
    if model_class == "nwc":
        model = SimpleVAECompressionModel(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_bn":
        model = SimpleVAECompressionModel_bn(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_ql":
        no_layernorm = getattr(opts, "no_layernorm", False)
        use_pe = getattr(opts, "use_pe", False)
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not no_layernorm),
            use_hyper = opts.use_hyper,
            pe = use_pe,
            )
    elif model_class == "nwc_ql_scale_cond":
        ql_scale_cond = getattr(opts, 'ql_scale_cond', False)
        assert ql_scale_cond == True
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not opts.no_layernorm),
            use_hyper = opts.use_hyper,
            pe = opts.use_pe,
            scale_cond = ql_scale_cond
            )
    elif model_class == "nwc_ql_compand":
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not opts.no_layernorm),
            use_hyper = opts.use_hyper,
            # pe = opts.use_pe,
            use_companding=True,
            learnable_s= opts.learnable_s
        )
    elif model_class == "nwc_ql_ltc":
        model = NWC_ql_LTC(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not opts.no_layernorm),
            lattice='BarnesWallUnitVol',
            N = opts.ltc_N,
        )
    elif model_class == "nwc_ql2":
        model = NWC_ql_learnable_scale(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == 'nwc_hp':
        model = MeanScaleHyperprior(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            N = opts.N,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_hess":
        model = SimpleVAECompressionModel_hess(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            R = opts.R,
            m = opts.m,
            scale=scale,
            shift=shift,
        )
    
    elif model_class == "nwc_tr":
        model = SimpleVAECompressionModel_with_Transformer(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_layer=opts.n_resblock, # 변수명 통일을 위해 쌓고 싶은 resblock 개수 = 쌓고싶은 transformer layer 개수로 일단 통일 -> 수정하고 싶으면 수정하샘
            M = opts.M,
            scale=scale,
            shift=shift,
            use_hyper = False
        )    
    
    elif model_class == "nwc_tr_with_hyp":
        model = SimpleVAECompressionModel_with_Transformer(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_layer=opts.n_resblock, 
            M = opts.M,
            scale=scale,
            shift=shift,
            use_hyper = True
        )    
    elif model_class == "nwc_ql_cdt":
        model = NWC_ql_conditional(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            C = opts.C,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_ql_ste":
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            mode='ste'
        )
    elif model_class == "nwc_ql_sga":
        model = NWC_ql_SGA(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_ql_sga_vbr":
        model = NWC_ql_SGA_Vbr(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_ql_pe":
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
            M = opts.M,
            scale=scale,
            shift=shift,
            pe = True,
            use_hyper=opts.use_hyper
        )
    elif model_class == "nwc_qmap":
        model = NWC_qmap(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_qmap2":
        model = NWC_qmap2(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_qmap3":
        model = NWC_qmap2(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc_scale_cond":
        if not hasattr(opts, 'pre_normalize'):
            setattr(opts, 'pre_normalize', False)
        if not hasattr(opts, 'normalize'):
            setattr(opts, 'normalize', False)
        model = NWC_scale_cond(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not opts.no_layernorm),
            use_hyper = opts.use_hyper,
            # pe = opts.use_pe,
            pre_normalize = opts.pre_normalize,
            normalize = opts.normalize
        )
    elif model_class == "nwc_scale_cond_v2":
        if not hasattr(opts, 'pre_normalize'):
            setattr(opts, 'pre_normalize', False)
        if not hasattr(opts, 'normalize'):
            setattr(opts, 'normalize', False)
        model = NWCScaleCond(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            # scale=scale,
            # shift=shift,
            norm = (not opts.no_layernorm),
            use_hyper = opts.use_hyper,
            pe = opts.use_pe,
            pre_normalize = opts.pre_normalize,
        )
    elif model_class == "nwc_scale_cond_ltc":
        model = NWC_scale_cond_ltc(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            M = opts.M,
            scale=scale,
            shift=shift,
            norm = (not opts.no_layernorm),
            # pe = opts.use_pe,
            # lattice='Leech2ProductUnitVol',
            lattice='Leech2UnitVol',
            N = opts.ltc_N,
        )
    else:
        raise
    return model

# elif model_class == "nwc_cdt2":
#         model = NWC_conditional2(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             n_resblock=opts.n_resblock,
#             Q = opts.Q,
#             C = opts.C,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "nwc_cdt_ln":
#         model = NWC_conditional_ln(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             n_resblock=opts.n_resblock,
#             Q = opts.Q,
#             C = opts.C,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "nwc_ql_bn":
#         model = NWC_ql_bn(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             n_resblock=opts.n_resblock,
#             Q = opts.Q,
#             scale=scale,
#             shift=shift,
#         )

# if model_class == "vq_seedlm":
#         model = VQ_SEEDLM(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             n_embeddings=opts.n_embeddings,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift
#             )
#     elif model_class == "vqvae":
#         assert opts.K is not None
#         model = VQVAE(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             dim_embeddings=opts.dim_embeddings,
#             n_embeddings=2**opts.K,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "vqvae_idx":
#         assert opts.K is not None
#         model = VQVAE_IDX(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             dim_embeddings=opts.dim_embeddings,
#             n_embeddings=opts.n_embeddings,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "vqvae_mag":
#         model = VQVAE_MAG(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             dim_embeddings=opts.dim_embeddings,
#             n_embeddings=opts.n_embeddings,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "vqvae_scale":
#         model = VQVAE_SCALE(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             dim_embeddings=opts.dim_embeddings,
#             n_embeddings=opts.n_embeddings,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift,
#         )
#     elif model_class == "vqvae_idx_mag":
#         model = VQVAE_IDX_MAG(
#             input_size=opts.input_size,
#             dim_encoder=opts.dim_encoder,
#             P=opts.P,
#             dim_embeddings=opts.dim_embeddings,
#             n_embeddings=opts.n_embeddings,
#             n_resblock=opts.n_resblock,
#             beta=opts.vq_beta,
#             scale=scale,
#             shift=shift,
#         )