import argparse
import os
from pathlib import Path
import json
import sys
import signal
import subprocess

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import math
import yaml

import utils
from Unified_BT_GOT_MLM_ITM_model import *
from datasets import *
from get_id_list import *
from get_id_list import get_id_list_separate
from itertools import chain
import pickle
from OT_torch_ import *

import torch.nn as nn
import torchvision
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Train Multimodal BT with GOT, MLM, and ITM')
parser.add_argument('--feature_dim', default=1024, type=int, help='Feature dim for latent vector')
parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
parser.add_argument('--num_workers', default=8, type=int, help='Number of parallel processing units')
parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--data_root_coco', default='/data/COCO/coco2014/', type=str, help='Root directory of the mscoco data')
parser.add_argument('--data_root_vg', default='/data/VG/', type=str, help='Root directory of the VG data')
parser.add_argument('--coco_version', default="2014", type=str, help="Version of mscoco data (2014/2017)")
parser.add_argument('--maxlen', default=30, type=int, help='Maximum length of tokens')
parser.add_argument('--text_model_name', default='roberta-base', type=str, help='Name of text model')
parser.add_argument('--projector_neurons', default=2048, type=int, help='Hidden dim of projector')

# for barlow twins
parser.add_argument('--lmbda_BT', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
parser.add_argument('--loss_name', default='BT', type=str, help='Name of Loss Function')

# for VICReg
parser.add_argument('--lmbda', default=25, type=int, help='Invariance loss coefficient')
parser.add_argument('--mu', default=25, type=int, help='Variance loss coefficient')
parser.add_argument('--nu', default=1, type=int, help='Covariance loss coefficient')

# for multi-modal loss
parser.add_argument('--alpha', default=0.25, type=float, help='Text1-Text2 loss coefficient')
parser.add_argument('--beta', default=0.25, type=float, help='Image1-Text1 loss coefficient')
parser.add_argument('--gamma', default=0.25, type=float, help='Image2-Text2 loss coefficient')

parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                             help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                             help='base learning rate for biases and batch norm parameters')
parser.add_argument('--got_lambda', default=0.1, type=float, help='hyperparameter for balancing WD and GWD loss in GOT')
parser.add_argument('--got_loss_weight', default=100.0, type=float, help='Weight for GOT_Loss')
parser.add_argument('--mlm_loss_weight', default=1.0, type=float, help='Weight for MLM_Loss')
parser.add_argument('--itm_loss_weight', default=1.0, type=float, help='Weight for ITM_Loss')
parser.add_argument('--checkpoint-dir', default='/checkpoint/ljng/multimodal/tmp', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--print_freq', default=50, type=int, help='frequency of steps to print loss')
parser.add_argument('--task_names', default='BTGOT, MLM, ITM', type=str, help='Task names for pre-training, includes BT, GOT, ITM and MLM')
parser.add_argument('--vg', action="store_true")
parser.add_argument('--name', type=str)


with open('./bt_got_mlm_itm_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 2 * len(loader)
    base_lr = args.batch_size / 512
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)

    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)



class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)
    def exclude_bias_and_norm(self, p):
        return p.ndim == 1
    @torch.no_grad()

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue
                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])



class BarlowTwinsGOT(nn.Module):
    def __init__(self, args, model_img_text):
        super().__init__()
        self.args = args
        self.model_img_text = model_img_text
        self.loss_name = args.loss_name
        self.lmbda_BT = args.lmbda_BT
        self.lmbda = args.lmbda
        self.mu = args.mu
        self.nu = args.nu
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.got_loss_weight = self.args.got_loss_weight
        self.mlm_loss_weight = self.args.mlm_loss_weight
        self.itm_loss_weight = self.args.itm_loss_weight
        self.bn = nn.BatchNorm1d(args.feature_dim, affine=False)


    def BarlowTwins(self,out_1,out_2,corr_neg_one=False):

        # empirical cross-correlation matrix
        # normalize the representations along the batch dimension

        c = self.bn(out_1).T @ self.bn(out_2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lmbda_BT * off_diag

        return loss


    def GOT(self, v_, q_):
        cos_distance = cost_matrix_batch_torch(v_.transpose(2, 1), q_.transpose(2, 1))
        cos_distance = cos_distance.transpose(1,2)
        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)

        wd = - IPOT_distance_torch_batch_uniform(cos_dist, v_.size(0), v_.size(1), q_.size(1), 30)
        torch.distributed.all_reduce(wd)
        gwd = GW_distance_uniform(v_.transpose(2,1), q_.transpose(2,1))
        torch.distributed.all_reduce(gwd)
        twd = self.args.got_lambda * torch.mean(gwd) + self.args.got_lambda * torch.mean(wd)

        return twd

    def compute_mlm(self, text_mlm_labels, infer, vocab_size=50265):
        mlm_logits = infer["cross_attn_mlm_logits"]
        mlm_labels = text_mlm_labels

        mlm_loss = torch.nn.functional.cross_entropy(
            mlm_logits.view(-1, vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

        torch.distributed.all_reduce(mlm_loss)

        return mlm_loss

    def compute_itm(self, infer):

        itm_logits = infer["cross_attn_itm_logits"]
        itm_labels = infer["cross_attn_itm_labels"]
        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

        torch.distributed.all_reduce(itm_loss)

        return itm_loss

    def forward(self, pos_im1, pos_im2, pos_tok1, pos_tok2, pos_mask1, pos_mask2, pos_mlm_tok1, pos_mlm_tok2, pos_mlm_label1, pos_mlm_label2, device=None, task_names = 'BTGOT, MLM, ITM'):
        ret_im_text_1 = self.model_img_text(pos_im1, pos_tok1, false_image_0=None, text_mlm_ids=pos_mlm_tok1, text_masks=pos_mask1, device=device)
        ret_im_text_2 = self.model_img_text(pos_im2, pos_tok2, false_image_0=None, text_mlm_ids=pos_mlm_tok2, text_masks=pos_mask2, device=device)

        loss_dict = {}

        # BTGOT
        loss_bt = (1 - self.alpha - self.beta - self.gamma) * self.BarlowTwins(ret_im_text_1["image_out"], ret_im_text_2["image_out"]) + self.alpha * self.BarlowTwins(ret_im_text_1["text_out"], ret_im_text_2["text_out"]) # intra BT
        loss_bt += self.beta * self.BarlowTwins(ret_im_text_1["image_out"], ret_im_text_1["text_out"]) + self.gamma * self.BarlowTwins(ret_im_text_2["image_out"], ret_im_text_2["text_out"]) # inter BT
        loss_got = self.GOT(ret_im_text_1["image_local_feats"], ret_im_text_1["text_local_feats"]) + self.GOT(ret_im_text_2["image_local_feats"], ret_im_text_2["text_local_feats"])
        loss = loss_bt + self.got_loss_weight * loss_got
        loss_dict.update({"loss_bt": loss_bt, "loss_got": loss_got})

        # MLM
        if 'MLM' in self.args.task_names:
            loss_mlm = 0.5 * (self.compute_mlm(pos_mlm_label1, ret_im_text_1, vocab_size=50265) + self.compute_mlm(pos_mlm_label2, ret_im_text_2, vocab_size=50265))
            loss = loss + self.mlm_loss_weight * loss_mlm
            loss_dict.update({"loss_mlm": loss_mlm})

        # ITM
        if 'ITM' in self.args.task_names:
            loss_itm = 0.5 * (self.compute_itm(ret_im_text_1) + self.compute_itm(ret_im_text_2))
            loss = loss + self.itm_loss_weight * loss_itm
            loss_dict.update({"loss_itm": loss_itm})

        loss_dict.update({"loss_total": loss})

        return loss_dict


def main_worker(gpu, args):

    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.checkpoint_dir = args.checkpoint_dir / args.name

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True


    feature_dim = args.feature_dim
    batch_size, epochs, loss_name = args.batch_size, args.epochs, args.loss_name
    lmbda_BT = args.lmbda_BT


    model = BarlowTwinsGOT(args=args, model_img_text=Model(config=config, feature_dim=feature_dim, projector_neurons=args.projector_neurons, text_model_name = args.text_model_name, maxlen = args.maxlen, task_names = args.task_names)).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)


    if args.coco_version=="2017":
        image_list_coco, text_list_coco = get_id_list_separate(args.data_root_coco)
    else:
        image_list_coco, text_list_coco = get_id_list_coco2014(args.data_root_coco) #Karpathy split of coco2014

    image_list_vg, text_list_vg = get_id_list_vg(args.data_root_vg) if args.vg else ([], [])

    image_list = image_list_coco + image_list_vg
    text_list = text_list_coco + text_list_vg

    cc_data = CCImageTextDataset(image_list = image_list, text_list = text_list, maxlen = args.maxlen, model_name = args.text_model_name)

    sampler = torch.utils.data.distributed.DistributedSampler(cc_data)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    train_loader = DataLoader(cc_data, batch_size=per_device_batch_size, num_workers=args.num_workers, sampler=sampler, pin_memory=True)

    if args.rank == 0:
        print("Dataloader Initialized")
        print("Model Initialized")

    # training loop
    results = {'BT_train_loss': [], 'GOT_train_loss': [], 'MLM_train_loss': [], 'ITM_train_loss': []}

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                                        map_location='cpu')
        model.module.model_img_text.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
    else:
        start_epoch = 0


    best_acc = 0.0

    if args.rank == 0:
        print("Starting training")

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for step, (image, token, mask, mlm_ids, mlm_labels) in enumerate(train_loader, start=epoch * len(train_loader)):
            pos_im1, pos_im2 = image[0], image[1]
            pos_tok1, pos_tok2 = token[0], token[1]
            pos_mask1, pos_mask2 = mask[0], mask[1]
            pos_mlm_tok1, pos_mlm_tok2 = mlm_ids[0], mlm_ids[1]
            pos_mlm_label1, pos_mlm_label2 = mlm_labels[0], mlm_labels[1]
            batch_size, _, _, _ = pos_im1.size()

            # BT and GOT
            pos_im1, pos_im2 = pos_im1.cuda(gpu, non_blocking=True), pos_im2.cuda(gpu, non_blocking=True)
            pos_tok1, pos_tok2 = pos_tok1.cuda(gpu, non_blocking=True), pos_tok2.cuda(gpu, non_blocking=True)
            pos_mask1, pos_mask2 = pos_mask1.cuda(gpu, non_blocking=True), pos_mask2.cuda(gpu, non_blocking=True)
            # ITM and MLM
            pos_mlm_tok1, pos_mlm_tok2 = pos_mlm_tok1.cuda(gpu, non_blocking=True), pos_mlm_tok2.cuda(gpu, non_blocking=True)
            pos_mlm_label1, pos_mlm_label2 = pos_mlm_label1.cuda(gpu, non_blocking=True), pos_mlm_label2.cuda(gpu, non_blocking=True)

            adjust_learning_rate(args, optimizer, train_loader, step)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss_dict = model(pos_im1, pos_im2, pos_tok1, pos_tok2, pos_mask1, pos_mask2, pos_mlm_tok1, pos_mlm_tok2, pos_mlm_label1, pos_mlm_label2, device=None, task_names=args.task_names)

            loss = loss_dict["loss_total"].mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            results['BT_train_loss'].append(loss_dict["loss_bt"].mean().item())
            results['GOT_train_loss'].append(loss_dict["loss_got"].mean().item())

            if 'MLM' in args.task_names:
                results['MLM_train_loss'].append(loss_dict["loss_mlm"].mean().item())
            if 'ITM' in args.task_names:
                results['ITM_train_loss'].append(loss_dict["loss_itm"].mean().item())

            if args.rank==0:
                if (step) % args.print_freq == 0:
                    stats = dict(epoch=epoch, step=step, total_epoch = epochs, steps_per_epoch=len(train_loader),
                            lr_weights=optimizer.param_groups[0]['lr'],
                            lr_biases=optimizer.param_groups[1]['lr'],
                            loss=loss.item(),
                            time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)


        if args.rank==0:
            state = dict(epoch=epoch + 1, model=model.module.model_img_text.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')


            with open(args.checkpoint_dir / str('mscoco_BT_loss_swin_' + str(args.text_model_name) + '_' + str(args.batch_size) + '_' + str(args.projector_neurons) + '.pkl'), 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    main()
