from .datasets_weight_block import get_datasets_block
from .datasets_weight_block_idx_calib_mag import get_dataset_block_16_calib, LayerInputs
from .datasets_weight_vector_idx_calib_mag import get_dataset_vector_calib, LayerInputs
from .datasets_weight_vector_idx_random_scale import get_dataset_vector_random_scale, LayerInputs
from .datasets_weight_block_qlevel import get_dataset_block_qlevel
from .datasets_weight_block_qlevel_random import get_dataset_block_random_qlevel
from .datasets_weight_block_seq import get_datasets_block_seq
from .datasets_weight_block_gaussian import get_datasets_gaussian_block
from .datasets_weight_block_seq_qlevel_random import get_datasets_block_seq_random_qlevel
from .datasets_weight_block_seq_gaussian import get_datasets_gaussian_block_seq


def get_datasets(opts):
    if opts.dataset == 'vector_mag':
        train_dataset, test_dataset, train_std, test_std = get_dataset_vector_calib(opts.block_direction)
    elif opts.dataset == 'vector_random_scale':
        train_dataset, test_dataset, train_std, test_std = get_dataset_vector_random_scale(opts.block_direction)
    elif opts.dataset == 'block':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block(opts.block_direction, opts.input_size)
    elif opts.dataset == 'block_mag':
        train_dataset, test_dataset, train_std, test_std = get_dataset_block_16_calib(opts.block_direction)
    elif opts.dataset == 'block_ql':
        train_dataset, test_dataset, train_std, test_std = get_dataset_block_qlevel(opts.block_direction)
    elif opts.dataset == 'block_ql_random':
        train_dataset, test_dataset, train_std, test_std = get_dataset_block_random_qlevel(opts.block_direction, opts.input_size)
    elif opts.dataset == 'block_seq':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq(opts.block_direction, opts.input_size)
    elif opts.dataset == 'gaussian':
        train_dataset, test_dataset, train_std, test_std = get_datasets_gaussian_block(opts.block_direction, opts.input_size)
    elif opts.dataset == 'block_seq_ql_random':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_qlevel(opts.block_direction, opts.input_size)
    elif opts.dataset == 'gaussian_seq':
        train_dataset, test_dataset, train_std, test_std = get_datasets_gaussian_block_seq(opts.block_direction, opts.input_size)
            
    
    return train_dataset, test_dataset, train_std, test_std