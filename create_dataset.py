import argparse
import numpy as np
from dataset.utils import create_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create artificially-noised LibriSpeech dataset.')

    # Data options
    parser.add_argument(
        '-s', '--start',
        type=int, default=0, required=True,
        help='first LibriSpeech index to use in dataset')
    parser.add_argument(
        '-e', '--end',
        type=int, default=0, required=True,
        help='last LibriSpeech index to use in dataset')
    parser.add_argument(
        '-iter', '--n-iters',
        type=int, default=1,
        help='number of iterations over each data sample')
    parser.add_argument(
        '--seed',
        type=int, default=1,
        help='seed value for random number generator')
    parser.add_argument(
        '-sr', '--srate',
        type=int, default=16000,
        help='audio signal sampling rate')
    # Data path arguments
    parser.add_argument(
        '-i', '--data-root',
        type=str, default='data/LibriSpeech/dev-clean',
        help='path to dataset root')
    parser.add_argument(
        '-o', '--out-dir',
        type=str, default='data/BabbledLibri',
        help='destination path to created dataset')
    # Boolean flag arguments
    parser.add_argument(
        '--test', action='store_true',
        help='dataset split is for testing')
    # Range arguments
    parser.add_argument(
        '-bg', '--babble-gain',
        type=float, nargs='+',
        default=[0.01, 0.05, 5.],
        help='babble gain range, args to np.linspace(lower, upper, n)')
    parser.add_argument(
        '-bs', '--babble-samples',
        type=int, nargs='+',
        default=[40, 71, 10],
        help=('number of samples constituting babble range, '
            'args to np.arange(lower, upper, step)'))
    parser.add_argument(
        '-rg', '--room-gain',
        type=float, nargs='+',
        default=[0.5, 2., 4],
        help='room gain range, args to np.linspace(lower, upper, n)')

    args = parser.parse_args()

    # Parse range arguments
    if (len(args.babble_gain) > 2):
        g_babble_range = np.linspace(
            args.babble_gain[0], args.babble_gain[1], int(args.babble_gain[2]))
    else:
        g_babble_range = np.linspace(0.01, 0.05, 5)

    if (len(args.room_gain) > 2):
        g_room_range = np.linspace(
            args.room_gain[0], args.room_gain[1], int(args.room_gain[2]))
    else:
        g_room_range = np.linspace(0.5, 2, 4)

    if (len(args.babble_samples) > 2):
        n_samples_babble_range = np.arange(*args.babble_samples)
    else:
        n_samples_babble_range = np.arange(40, 71, 10)

    create_dataset(
        args.data_root,
        args.out_dir,
        start_idx=args.start,
        end_idx=args.end,
        g_babble_range=g_babble_range,
        g_room_range=g_room_range,
        n_samples_babble_range=n_samples_babble_range,
        n_iters=args.n_iters,
        test=args.test,
        seed=args.seed,
        srate=args.srate)
