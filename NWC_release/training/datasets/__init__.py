from .datasets_weight_block_seq_qlevel_random import get_datasets_block_seq_random_qlevel
from .datasets_weight_block_seq_hesseigen_random import get_datasets_block_seq_random_hesseigen
from .datasets_weight_block_seq import get_datasets_block_seq
from .datasets_weight_block_seq_qlevel_random_lstats import get_datasets_block_seq_random_qlevel_lstats
from .datasets_weight_vector import get_datasets_vector
from .datasets_weight_vector_qlevel_random import get_datasets_vector_random_qlevel
from .datasets_weight_block_seq_qmap_uniform import get_datasets_block_seq_qmap_uniform
from .datasets_weight_block_seq_scale_cond import get_datasets_block_seq_scale_cond
def get_datasets(args):
    if args.dataset == 'block_seq_ql_random':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_qlevel(args.dataset_path, args.input_size, args.Q, args)
    elif args.dataset == 'block_seq_ql_random_lstats':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_qlevel_lstats(args.dataset_path, args.input_size, args.Q, args)
    elif args.dataset == 'block_seq_hesseigen':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_hesseigen(args.dataset_path, args.input_size, args.R, args)
    elif args.dataset == 'block_seq':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq(args.dataset_path, args.input_size, args)
    elif args.dataset == 'block_vec':
        train_dataset, test_dataset, train_std, test_std = get_datasets_vector(args.dataset_path, args)
    elif args.dataset == 'block_vec_ql_random':
        train_dataset, test_dataset, train_std, test_std = get_datasets_vector_random_qlevel(args.dataset_path, args.Q, args)
    elif args.dataset == 'block_seq_ql_random_pos':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_qlevel(args.dataset_path, args.input_size, args.Q, args, return_idx_ltype = True)
    elif args.dataset == 'block_seq_qmap':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_qmap_uniform(args.dataset_path, args.input_size, args)
    elif args.dataset == 'block_seq_scale_cond':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_scale_cond(args.dataset_path, args.input_size, args)
    elif args.dataset.startswith('block_seq_scale_cond_uniform'):
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_scale_cond(args.dataset_path, args.input_size, args, True, args.uniform_scale_max)
    return train_dataset, test_dataset, train_std, test_std