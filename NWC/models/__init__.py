from .vqvae import VQVAE
from .vqvae_idx import VQVAE_IDX
from .vqvae_calib_mag import VQVAE_MAG
from .vqvae_scale import VQVAE_SCALE
from .vqvae_idx_mag import VQVAE_IDX_MAG
from .nwc import SimpleVAECompressionModel, MeanScaleHyperprior, SimpleVAECompressionModel_with_Transformer
from .nwc_ql import NWC_ql
from .nwc_ql_cdt import NWC_ql_conditional
# from .nwc_ql_cdt import NWC_conditional, NWC_conditional2
# from .nwc_ql_cdt_ln import NWC_conditional_ln
# from .nwc_hess import SimpleVAECompressionModel_hess
# from .nwc_ql_batchnorm import NWC_ql_bn
# from .nwc_bn import SimpleVAECompressionModel_bn

def get_model(model_class, opts, scale, shift):
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
        model = NWC_ql(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            n_resblock=opts.n_resblock,
            Q = opts.Q,
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