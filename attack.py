from datasets import DATASETS
from config import STATE_DICT_KEY

from model import *
from adversarial import *
from dataloader import *
from trainer import *
from utils import *

import argparse
import torch
import copy
from pathlib import Path
from collections import defaultdict


def attack(args, attack_item_num=2, model_root=None):
    fix_random_seed_as(args.model_init_seed)
    _, _, test_loader = dataloader_factory(args)

    if args.model_code == 'locker':
        model = Locker(args)
    elif args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)

    if args.defense_method is not None:
        print('Attack {} defense models...'.format(args.defense_method))
        model_root = 'experiments/' + args.defense_method + '/' + args.model_code + '/' + args.dataset_code
    else:
        print('Attack normal models...')
        model_root = 'experiments/' + args.model_code + '/' + args.dataset_code
    
    model.load_state_dict(torch.load(os.path.join(model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))

    item_counter = defaultdict(int)
    dataset = dataset_factory(args)
    dataset = dataset.load_dataset()
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    for user in train.keys():
        seqs = train[user] + val[user] + test[user]
        for i in seqs:
            item_counter[i] += 1

    item_popularity = []
    for i in item_counter.keys():
        item_popularity.append((item_counter[i], i))
    item_popularity.sort(reverse=True)
    
    print('***** Untargeted Substitution Attacks *****')
    attacker = AdversarialAttacker(args, model, test_loader)
    metrics_before = attacker.test()
    metrics_ours = attacker.substitution_attack(num_attack=attack_item_num)
    
    test_items = 15
    test_split = test_items // 3
    step = len(item_popularity) // test_items
    attack_ranks = list(range(0, len(item_popularity), step))[:test_items]

    print('***** Targeted Substitution Attacks *****')
    for i in attack_ranks:
        item = item_popularity[i][1]
        metrics_before = attacker.test(target=item)
        metrics_ours = attacker.substitution_attack(target=item, num_attack=attack_item_num)


if __name__ == "__main__":
    set_template(args)
    if args.dataset_code == 'ml-1m':
        attack(args=args, attack_item_num=2)
    else:
        attack(args=args, attack_item_num=1)
