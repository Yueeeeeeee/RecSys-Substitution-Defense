from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
import numpy as np
from abc import *
from pathlib import Path


class SASAdvTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)
        
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

        self.bce = nn.BCEWithLogitsLoss()

        if self.args.defense_method == 'dirichlet':
            self.build_joint_synonym_matrix()
            self.build_dirichlet_coeff_cache()
            self.train_loader.collate_fn = self.expand_and_sample_dirichlet

    def build_joint_synonym_dict(self, min_items=2, max_items=50):
        print('Building synonym dict...')
        self.build_1_order_synonym_dict(min_items=min_items, max_items=max_items//2)
        self.build_2_order_synonym_dict(max_items=max_items//2)
        self.joint_synonym_dict = {}
        for key, value in self.synonym_dict_1_order.items():
            self.joint_synonym_dict[key] = [value]
        for key, value in self.synonym_dict_2_order.items():
            self.joint_synonym_dict[key] += [value]
    
    def build_joint_synonym_matrix(self, min_items=2, max_items=50):
        print('Building synonym matrix...')
        self.build_1_order_synonym_dict(min_items=min_items, max_items=max_items//2)
        self.build_2_order_synonym_dict(max_items=max_items//2)
        self.lengths_1_order = np.zeros(len(self.synonym_dict_1_order)+1).astype(int)
        self.lengths_2_order = np.zeros(len(self.synonym_dict_2_order)+1).astype(int)
        self.joint_synonym_matrix = np.zeros(
            (len(self.synonym_dict_1_order)+1, max_items))
        for key, value in self.synonym_dict_1_order.items():
            self.lengths_1_order[key] = len(value)
            self.joint_synonym_matrix[key, :len(value)] = value
        for key, value in self.synonym_dict_2_order.items():
            self.lengths_2_order[key] = len(value)
            start_pos = self.lengths_1_order[key]
            self.joint_synonym_matrix[key, start_pos:start_pos+len(value)] = value
        self.joint_synonym_matrix = torch.tensor(self.joint_synonym_matrix).long()

    def build_1_order_synonym_dict(self, min_items=2, max_items=25):
        self.synonym_dict_1_order = {}
        self.model.eval()
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            item_embeddings = self.model.embedding.token.weight.detach()[1:]
            for i in range(len(item_embeddings)):
                item_id = i + 1  # item id starts from 1
                item_vec = item_embeddings[i]
                scores = cos(item_vec, item_embeddings)
                indices = (scores >= self.args.min_cos_sim).nonzero(as_tuple=False)
                if len(indices) <= min_items:
                    _, indices = torch.sort(scores, descending=True)
                    indices = indices[:(min_items+1)]
                synonym_items = (indices+1).squeeze().cpu().numpy().tolist()

                if item_id in synonym_items:
                    synonym_items.remove(item_id)
                if len(synonym_items) > max_items:
                    synonym_items = np.random.choice(
                        synonym_items, max_items, replace=False).tolist()
                self.synonym_dict_1_order[item_id] = synonym_items
    
    def build_2_order_synonym_dict(self, max_items=25):
        self.synonym_dict_2_order = {}
        for item_id in self.synonym_dict_1_order.keys():
            synonym_items_2_order = []
            synonyms = self.synonym_dict_1_order[item_id]
            for synonym in synonyms:
                synonym_items_2_order += self.synonym_dict_1_order[synonym]
            synonym_items_2_order = set(synonym_items_2_order)
            if item_id in synonym_items_2_order:
                synonym_items_2_order.remove(item_id)
            
            for synonym in synonyms:
                if synonym in synonym_items_2_order:
                    synonym_items_2_order.remove(synonym)
            if len(synonym_items_2_order) > max_items:
                synonym_items_2_order = list(synonym_items_2_order)
                synonym_items_2_order = np.random.choice(
                    synonym_items_2_order, max_items, replace=False).tolist()
            self.synonym_dict_2_order[item_id] = list(synonym_items_2_order)
    
    def build_synonym_swap_cache(self):
        print('Building synonym cache...')
        self.synonym_cache = {}
        for item in self.joint_synonym_dict.keys():
            cur_synonyms = self.joint_synonym_dict[item]
            probs = [self.args.alpha] * len(cur_synonyms[0]) + \
                [self.args.decay*self.args.decay] * len(cur_synonyms[1])
            probs = np.array(probs) / sum(probs)
            self.synonym_cache[item] = np.random.choice(
                cur_synonyms[0]+cur_synonyms[1], size=self.args.cache_size, p=probs)

    def sample_synonym_fast(self, item, idx):
        return self.synonym_cache[item][idx % self.args.cache_size]

    def build_dirichlet_coeff_cache(self):
        print('Building dirichlet cache...')
        self.dirichlet_cache = {}
        for num_synonym_1_order in np.unique(self.lengths_1_order):
            if num_synonym_1_order == 0:
                continue
            for num_synonym_2_order in np.unique(self.lengths_2_order):
                alphas = [self.args.alpha] * num_synonym_1_order + \
                    [self.args.alpha * self.args.decay] * num_synonym_2_order
                dirichlet = np.random.dirichlet(alphas, self.args.cache_size**2).astype(np.float32)
                zeros = np.zeros((self.args.cache_size**2,
                    self.joint_synonym_matrix.shape[1]-dirichlet.shape[1])).astype(np.float32)
                self.dirichlet_cache[(num_synonym_1_order, num_synonym_2_order)] = \
                    torch.tensor(np.concatenate((dirichlet, zeros), axis=1))

    def sample_dirichlet_synonyms_fast(self, item, idx, require_coeff=True):
        synonyms = self.joint_synonym_matrix[item]
        num_synonym_1_order = self.lengths_1_order[item]
        num_synonym_2_order = self.lengths_2_order[item]
        num_synonym = num_synonym_1_order + num_synonym_2_order
        coeffs = None
        if require_coeff:
            coeffs = self.dirichlet_cache[(num_synonym_1_order, \
                num_synonym_2_order)][idx%(self.args.cache_size**2)] 
        return num_synonym, synonyms, coeffs

    def expand_and_sample_dirichlet(self, batch):
        seqs, labels, negs = [], [], []
        for i in range(len(batch)):
            seq = batch[i][0]
            seq_expanded = F.one_hot(seq,
                num_classes=self.model.embedding.token.weight.size(0)).float()
            item_positions = seq.nonzero().squeeze().view(-1)
            min_sample = max(1, int(self.args.substitution_ratio*len(item_positions)))
            swap_positions = np.random.choice(item_positions, min_sample, replace=False)
            seq_expanded[swap_positions, :] = 0.
            for j in swap_positions:
                num_synonym, synonyms, coeffs = self.sample_dirichlet_synonyms_fast(seq[j].item(), i*j)
                seq_expanded[j, synonyms[:num_synonym]] = coeffs[:num_synonym]
            seqs.append(seq_expanded)
            labels.append(batch[i][1])
            negs.append(batch[i][2])
        return torch.stack(seqs), torch.stack(labels), torch.stack(negs)

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            if (epoch+1) % self.args.dict_update_epoch == 0:
                if self.args.defense_method == 'dirichlet':
                    self.build_joint_synonym_matrix()
                    self.build_dirichlet_coeff_cache()
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            loss_val = loss.item()

            self.clip_gradients(5)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            if self.args.defense_method == 'advtrain':
                seq_before = batch[0].clone()
                batch[0] = F.one_hot(batch[0],
                    num_classes=self.model.embedding.token.weight.size(0)).float()
                
                self.model.eval()
                with torch.enable_grad():
                    for _ in range(self.args.adv_iteration):
                        batch[0].requires_grad = True
                        loss = self.calculate_loss(batch)
                        loss.backward()
                        input_grad = batch[0].grad.data / \
                            (torch.norm(batch[0].grad.data, dim=-1, keepdim=True) + 1e-9)
                        # also possible to normalize with log and exp
                        # batch[0] = (batch[0] + 1e-9).log() + self.args.adv_step * input_grad
                        # batch[0] = F.softmax(batch[0], dim=-1).detach()
                        batch[0] = batch[0] + self.args.adv_step * input_grad
                        batch[0] = torch.clamp(batch[0], min=0.)
                        batch[0] = batch[0] / batch[0].sum(-1, keepdim=True)
                        batch[0] = batch[0].detach()

                switch_indices = (torch.rand(seq_before.shape) <= self.args.substitution_ratio).to(self.device) 
                switch_indices = (switch_indices * (seq_before != 0)).float().unsqueeze(-1)
                batch[0] = switch_indices * batch[0] + (1 - switch_indices) * F.one_hot(seq_before,
                    num_classes=self.model.embedding.token.weight.size(0)).float()
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.calculate_loss(batch)
                loss.backward()
                loss_val += loss.item()

                self.clip_gradients(5)
                self.optimizer.step()

            average_meter_set.update('loss', loss_val)
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

    def test(self):
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        all_scores = []
        average_scores = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)
                
                # seqs, candidates, labels = batch
                # scores = self.model(seqs)
                # scores = scores[:, -1, :]
                # scores_sorted, indices = torch.sort(scores, dim=-1, descending=True)
                # all_scores += scores_sorted[:, :100].cpu().numpy().tolist()
                # average_scores += scores_sorted.cpu().numpy().tolist()
                # scores = scores.gather(1, candidates)
                # metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics

    def calculate_loss(self, batch):
        seqs, labels, negs = batch

        logits = self.model(seqs)
        if len(seqs.shape) > 2:
            pos_logits = logits.gather(-1, labels.unsqueeze(-1))[seqs[:, :, 1:].sum(-1) > 0].squeeze()
            neg_logits = logits.gather(-1, negs.unsqueeze(-1))[seqs[:, :, 1:].sum(-1) > 0].squeeze()
        else:
            pos_logits = logits.gather(-1, labels.unsqueeze(-1))[seqs > 0].squeeze()
            neg_logits = logits.gather(-1, negs.unsqueeze(-1))[seqs > 0].squeeze()
            
        pos_targets = torch.ones_like(pos_logits)
        neg_targets = torch.zeros_like(neg_logits)
        loss = self.bce(torch.cat((pos_logits, neg_logits), 0), torch.cat((pos_targets, neg_targets), 0))
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch

        scores = self.model(seqs)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
