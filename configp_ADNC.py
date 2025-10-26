import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AD_dir', type=str,
                        help='subfloder of train or test dataset', default='AD/')
    parser.add_argument('--CN_dir', type=str,
                        help='subfloder of train or test dataset', default='CN/')
    parser.add_argument('--MCI_dir', type=str,
                        help='subfloder of train or test dataset', default='MCI/')
    parser.add_argument('--PMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='PMCI/')
    parser.add_argument('--SMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='SMCI/')

    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.0001)')
    parser.add_argument('--class_num', type=int, help='class_num', default=2)
    parser.add_argument('--seed', type=int, help='Seed', default=42)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
    parser.add_argument('--train_root_path', type=str, help='Root path for train dataset',
                        default='')
    parser.add_argument('--val_root_path', type=str, help='Root path for val dataset',
                        default='')
    parser.add_argument('--test_root_path', type=str, help='Root path for test dataset',
                        default='')
    parser.add_argument('--batch_size', type=int, help='batch_size of data', default=12)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=69)
    return parser.parse_args()
