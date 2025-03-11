from .vqvae import VQVAE
from .vqvae_idx import VQVAE_IDX
from .vqvae_calib_mag import VQVAE_MAG
from .vqvae_scale import VQVAE_SCALE
from .vqvae_idx_mag import VQVAE_IDX_MAG
from .nwc import SimpleVAECompressionModel, MeanScaleHyperprior
from .nwc_ql import NWC_ql

def get_model(model_class, opts, scale, shift):
    if model_class == "vq_seedlm":
        model = VQ_SEEDLM(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            n_embeddings=opts.n_embeddings,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift
            )
    elif model_class == "vqvae":
        assert opts.K is not None
        model = VQVAE(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            dim_embeddings=opts.dim_embeddings,
            n_embeddings=2**opts.K,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift,
        )
    elif model_class == "vqvae_idx":
        assert opts.K is not None
        model = VQVAE_IDX(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            dim_embeddings=opts.dim_embeddings,
            n_embeddings=opts.n_embeddings,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift,
        )
    elif model_class == "vqvae_mag":
        model = VQVAE_MAG(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            dim_embeddings=opts.dim_embeddings,
            n_embeddings=opts.n_embeddings,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift,
        )
    elif model_class == "vqvae_scale":
        model = VQVAE_SCALE(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            dim_embeddings=opts.dim_embeddings,
            n_embeddings=opts.n_embeddings,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift,
        )
    elif model_class == "vqvae_idx_mag":
        model = VQVAE_IDX_MAG(
            input_size=opts.input_size,
            dim_encoder=opts.dim_encoder,
            P=opts.P,
            dim_embeddings=opts.dim_embeddings,
            n_embeddings=opts.n_embeddings,
            n_resblock=opts.n_resblock,
            beta=opts.vq_beta,
            scale=scale,
            shift=shift,
        )
    elif model_class == "nwc":
        model = SimpleVAECompressionModel(
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
    else:
        raise
    return model