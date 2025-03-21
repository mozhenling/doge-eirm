
#-- future
from datautils.diag_datasets import DATASETS as diag_DATASETS
from datautils.bed_datasets import DATASETS as bed_DATASETS
def get_dataset_object(args, hparams):
    if args.dataset in diag_DATASETS:
        from datautils import diag_datasets as datasets_now
        return vars(datasets_now)[args.dataset](args.data_dir, args.device, args.test_envs,
                                                   args.label_noise_type, args.label_noise_rate)

    elif args.dataset in  bed_DATASETS:
        from datautils import bed_datasets as datasets_now
        return vars(datasets_now)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError('The dataset name is NOT found!')

def datasets_now(args):
    if args.dataset in diag_DATASETS:
        from datautils import diag_datasets as datasets
        return datasets

    elif args.dataset in  bed_DATASETS:
        from datautils import bed_datasets as datasets
        return datasets

    else:
        raise NotImplementedError('The dataset name is NOT found!')