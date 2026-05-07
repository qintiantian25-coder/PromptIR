import argparse
import configparser
import subprocess
import shlex
import os


def build_train_cmd(cfg):
    # map config to train.py CLI args
    args = ['python', 'train.py']
    train_cfg = cfg['train'] if 'train' in cfg else {}
    paths = cfg['paths'] if 'paths' in cfg else {}
    val_cfg = cfg['validation'] if 'validation' in cfg else {}

    def add_flag(name, val):
        if val is None:
            return
        args.extend([f'--{name}', str(val)])

    add_flag('cuda', train_cfg.get('cuda', None))
    add_flag('epochs', train_cfg.get('epochs', None))
    add_flag('batch_size', train_cfg.get('batch_size', None))
    add_flag('num_gpus', train_cfg.get('num_gpus', None))
    add_flag('dataset_path', paths.get('dataset_path', None))
    add_flag('ckpt_dir', paths.get('ckpt_dir', None))
    add_flag('output_path', paths.get('output_path', None))
    # use_blind_pairs is boolean; pass literal string
    ub = train_cfg.get('use_blind_pairs', None)
    if ub is not None:
        args.extend(['--use_blind_pairs', str(ub)])

    # additional optional flags
    if 'wblogger' in train_cfg:
        add_flag('wblogger', train_cfg.get('wblogger'))

    # validation flags
    add_flag('val_interval', val_cfg.get('val_interval', None))
    add_flag('val_blur_dir', val_cfg.get('val_blur_dir', None))
    add_flag('val_sharp_dir', val_cfg.get('val_sharp_dir', None))
    add_flag('best_model_path', val_cfg.get('best_model_path', None))

    return args


def build_test_cmd(cfg):
    args = ['python', 'test.py']
    test_cfg = cfg['test'] if 'test' in cfg else {}
    paths = cfg['paths'] if 'paths' in cfg else {}

    def add_flag(name, val):
        if val is None:
            return
        args.extend([f'--{name}', str(val)])

    add_flag('cuda', test_cfg.get('cuda', None))
    add_flag('mode', test_cfg.get('mode', None))
    add_flag('dataset_path', paths.get('dataset_path', None))
    add_flag('blind_dataset_path', paths.get('blind_dataset_path', None))
    add_flag('ckpt_name', test_cfg.get('ckpt_name', None))
    add_flag('output_path', paths.get('output_path', None))

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, default='experiment.cfg')
    parser.add_argument('--dry_run', action='store_true', help='Print command and exit')
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print('Config not found:', args.config_path)
        return

    cfgp = configparser.ConfigParser()
    cfgp.read(args.config_path)

    if args.train:
        cmd = build_train_cmd(cfgp)
        print('Running training command:')
        print(' '.join(shlex.quote(x) for x in cmd))
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
    elif args.test:
        cmd = build_test_cmd(cfgp)
        print('Running test command:')
        print(' '.join(shlex.quote(x) for x in cmd))
        if args.dry_run:
            return
        subprocess.run(cmd, check=True)
    else:
        print('Specify --train or --test')


if __name__ == '__main__':
    main()
