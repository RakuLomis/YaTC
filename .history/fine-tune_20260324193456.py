import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os
import PIL

from torchvision import datasets, transforms
from tqdm import tqdm

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_YaTC

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('YaTC fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='TraFormer_YaTC', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
#20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', default='./output_dir/pretrained-model.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str,
                        help='dataset path')
    parser.add_argument('--train_split', default='train', type=str,
                        help='train split folder name under data_path')
    parser.add_argument('--val_split', default='val', type=str,
                        help='validation split folder name under data_path')
    parser.add_argument('--test_split', default='test', type=str,
                        help='test split folder name under data_path')
    parser.add_argument('--eval_split', default='test', type=str,
                        help='which split to evaluate when using --eval')
    parser.add_argument('--save_eval_dir', default='', type=str,
                        help='directory to save eval metrics/confusion matrix when using --eval')
    parser.add_argument('--save_eval_prefix', default='eval', type=str,
                        help='file prefix for saved eval artifacts')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only (requires --resume)')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def build_dataset(split_name, args, required=True):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(args.data_path, split_name)
    if not os.path.isdir(root):
        if required:
            raise FileNotFoundError(f"Split folder not found: {root}")
        return None
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.eval and not args.resume:
        raise ValueError("--eval requires --resume to load a trained checkpoint.")

    dataset_train = None
    dataset_val = None
    dataset_eval = None

    if args.eval:
        dataset_eval = build_dataset(split_name=args.eval_split, args=args, required=True)
    else:
        dataset_train = build_dataset(split_name=args.train_split, args=args, required=True)
        dataset_val = build_dataset(split_name=args.val_split, args=args, required=False)
        if dataset_val is None:
            print(f"Validation split '{args.val_split}' not found, fallback to '{args.test_split}'.")
            dataset_val = build_dataset(split_name=args.test_split, args=args, required=True)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if not args.eval:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = None
        eval_dataset = dataset_eval if args.eval else dataset_val
        if args.dist_eval:
            if len(eval_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_eval = torch.utils.data.DistributedSampler(
                eval_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_eval = torch.utils.data.SequentialSampler(eval_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train) if not args.eval else None
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval if args.eval else dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval and SummaryWriter is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
        if global_rank == 0 and not args.eval and SummaryWriter is None:
            print("TensorBoard is not installed. Continue training without SummaryWriter.")

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_train = None

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval if args.eval else dataset_val, sampler=sampler_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        eval_stats = evaluate(data_loader_eval, model, device)
        eval_size = len(dataset_eval)
        print(f"Accuracy on {args.eval_split} ({eval_size} images): {eval_stats['acc1']:.4f}")
        print(f"F1 on {args.eval_split} ({eval_size} images): {eval_stats['macro_f1']:.4f}")
        if args.save_eval_dir:
            os.makedirs(args.save_eval_dir, exist_ok=True)
            cm = np.array(eval_stats['cm'])
            cm_csv_path = os.path.join(args.save_eval_dir, f"{args.save_eval_prefix}_cm.csv")
            np.savetxt(cm_csv_path, cm, fmt='%d', delimiter=',')

            metrics = {
                "split": args.eval_split,
                "num_images": eval_size,
                "acc": float(eval_stats['acc1']),
                "precision": float(eval_stats['macro_pre']),
                "recall": float(eval_stats['macro_rec']),
                "f1": float(eval_stats['macro_f1']),
                "loss": float(eval_stats['loss']),
                "cm_csv": cm_csv_path,
            }
            metrics_path = os.path.join(args.save_eval_dir, f"{args.save_eval_prefix}_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Saved eval confusion matrix to: {cm_csv_path}")
            print(f"Saved eval metrics to: {metrics_path}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0

    epoch_iter = tqdm(
        range(args.start_epoch, args.epochs),
        desc=f"Training epochs [{Path(args.data_path).name}]",
        unit="epoch",
        disable=not misc.is_main_process(),
    )

    for epoch in epoch_iter:
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats = evaluate(data_loader_eval, model, device)
        is_best_f1 = val_stats["macro_f1"] > max_f1

        print(f"Accuracy on {args.val_split} ({len(dataset_val)} images): {val_stats['acc1']:.4f}")
        max_accuracy = max(max_accuracy, val_stats["acc1"])
        print(f"F1 on {args.val_split} ({len(dataset_val)} images): {val_stats['macro_f1']:.4f}")
        max_f1 = max(max_f1, val_stats["macro_f1"])
        print(f'Max Accuracy: {max_accuracy:.4f}')
        print(f'Max F1: {max_f1:.4f}')
        if misc.is_main_process():
            epoch_iter.set_postfix(
                val_acc=f"{val_stats['acc1']:.4f}",
                val_f1=f"{val_stats['macro_f1']:.4f}",
                best_f1=f"{max_f1:.4f}"
            )

        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                name='last'
            )
            if is_best_f1:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    name='best'
                )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log_finetune.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
