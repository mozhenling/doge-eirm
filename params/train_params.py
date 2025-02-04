
import argparse

def get_args():
    """
    https://docs.python.org/3/library/argparse.html#
    """
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="ColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--nets_base', type=str, default="diag_nets",
                        help='networks for featurizer and classifier')
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--erm_loss', type=str, default='CELoss')
    parser.add_argument('--optimizer', type=str, default = 'Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler', type=str, default=None, choices=[None,'cos', 'lambda'])
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default=r'.\outputs\train_outs')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")

    parser.add_argument('--label_noise_type', type=str, default='nsfree', choices=['nsfree', 'sym', 'asym'])
    parser.add_argument('--label_noise_rate', type=float, default=0.) # corrupted the label by flipping
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')


    args = parser.parse_args()
    return args