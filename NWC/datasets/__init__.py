from .datasets_weight_block_seq_qlevel_random import get_datasets_block_seq_random_qlevel
from .datasets_weight_block_seq_hesseigen_random import get_datasets_block_seq_random_hesseigen
from .datasets_weight_block_seq import get_datasets_block_seq
# from .datasets_weight_block_seq_scalar_mean import get_datasets_block_seq_scalar_mean

def get_datasets(args):
    if args.dataset == 'block_seq_ql_random':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_qlevel(args.dataset_path, args.input_size, args.Q, args)
    elif args.dataset == 'block_seq_hesseigen':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq_random_hesseigen(args.dataset_path, args.input_size, args.R, args)
    elif args.dataset == 'block_seq':
        train_dataset, test_dataset, train_std, test_std = get_datasets_block_seq(args.dataset_path, args.input_size, args)
    return train_dataset, test_dataset, train_std, test_std