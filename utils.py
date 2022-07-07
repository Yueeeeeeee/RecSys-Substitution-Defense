import numpy as np
import random
import torch

from datasets import DATASETS
from model import *
import argparse


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_template(args):
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    dataset_code = {'1': 'ml-1m', '20': 'ml-20m', 'b': 'beauty' , 'g': 'games', 'l': 'lastfm1k', 's': 'steam', 'y': 'yoochoose'}
    args.dataset_code = dataset_code[input('Input 1 / 20 for movielens, b for beauty, g for games, l for lastfm, s for steam and y for yoochoose: ')]
    if args.dataset_code == 'ml-1m':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 40
    elif args.dataset_code in ['ml-20m', 'lastfm1k']:
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 50
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 20
    elif args.dataset_code == 'beauty':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.6
        args.bert_max_predictions = 30
    elif args.dataset_code == 'steam':
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20
    else:
        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.5
        args.bert_max_len = 50
        args.bert_mask_prob = 0.5
        args.bert_max_predictions = 25
    
    batch = 64
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765

    model_codes = {'b': 'bert', 'l': 'locker', 'n':'narm', 's':'sas'}
    args.model_code = model_codes[input('Input model code, b for BERT, l for Locker, n for NARM and s for SASRec: ')]

    if torch.cuda.is_available():
        args.device = 'cuda:' + input('Input GPU ID: ')
    else:
        args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = True
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100
    args.num_epochs = 1000

    args.metric_ks = [1, 5, 10]
    args.best_metric = 'NDCG@10'
    args.model_init_seed = 98765
    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None

    if args.model_code == 'locker':
        args.bert_num_heads = 4
        args.locker_l_heads = 3
        args.locker_kernel_size = 3


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--dataset_split_seed', type=int, default=0)

################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--sliding_window_size', type=float, default=0.5)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--train_negative_sample_size', type=int, default=0)
parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=0)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
# optimizer & lr#
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
# epochs #
parser.add_argument('--num_epochs', type=int, default=100)
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=['locker', 'bert', 'sas', 'narm'])
# BERT specs, used for SASRec and NARM as well #
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.1)
parser.add_argument('--bert_attn_dropout', type=float, default=0.1)
parser.add_argument('--bert_mask_prob', type=float, default=0.2)
# Locker specs (CNN version)
parser.add_argument('--locker_l_heads', type=int, default=3)
parser.add_argument('--locker_kernel_size', type=int, default=3)

################
# Attack & Defense
################
parser.add_argument('--num_attack_items', type=int, default=5)
parser.add_argument('--min_cos_sim', type=float, default=0.5)
parser.add_argument('--defense_method', type=str, default=None)
parser.add_argument('--num_attack_eval', type=int, default=8192)
parser.add_argument('--substitution_ratio', type=float, default=0.5)
parser.add_argument('--dict_update_epoch', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--decay', type=float, default=0.5)
parser.add_argument('--adv_iteration', type=int, default=3)
parser.add_argument('--adv_step', type=float, default=1.0)
parser.add_argument('--cache_size', type=int, default=100)

################


args = parser.parse_args()
