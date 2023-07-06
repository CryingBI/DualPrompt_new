# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from copy import deepcopy
import utils



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train_task_model(task_model: torch.nn.Module, device, gm_list, epochs, task_id=-1,):

    task_model.train()

    #training_data = [[] for e_id in range(epochs)]
    data_loader_data = []
    lr = 5e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(task_model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=90)

    gm_use = gm_list[:10*(task_id+1)]
    input_train = []
    target_train = []
    for i in range(len(gm_use)):
        input, _ = gm_use[i].sample(n_samples=128*90)
        input = torch.from_numpy(input).float()
        #target = torch.from_numpy(target).long()
        if i < 10:
            new_target = torch.Tensor([0]).expand(128*90).long()           #500
        elif i >= 10 and i < 20:
            new_target = torch.Tensor([1]).expand(128*90).long()           #556
        elif i >= 20 and i < 30:
            new_target = torch.Tensor([2]).expand(128*90).long()           #625
        elif i >= 30 and i < 40:
            new_target = torch.Tensor([3]).expand(128*90).long()           #714
        elif i >= 40 and i < 50:
            new_target = torch.Tensor([4]).expand(128*90).long()           #833
        elif i >= 50 and i < 60:
            new_target = torch.Tensor([5]).expand(128*90).long()          #128*900
        elif i >= 60 and i < 70:
            new_target = torch.Tensor([6]).expand(128*90).long()          #1250
        elif i >= 70 and i < 80:
            new_target = torch.Tensor([7]).expand(128*90).long()          #1666
        elif i >= 80 and i < 90:
            new_target = torch.Tensor([8]).expand(128*90).long()          #2500
        elif i >= 90 and i < 100:
            new_target = torch.Tensor([9]).expand(128*90).long()          #5000
        input_train.append(input)
        target_train.append(new_target)

    input_train = torch.cat(input_train, dim=0)
    target_train = torch.cat(target_train, dim=0)
    
    for e_id in range(epochs):

        input_train_raw = []
        target_train_raw = []
        for i in range(len(gm_use)):
            input_raw, target_raw = input_train[128*e_id+128*90*i : 128*(e_id+1)+128*90*i], target_train[128*e_id+128*90*i : 128*(e_id+1)+128*90*i]
            input_train_raw.append(input_raw)
            target_train_raw.append(target_raw)
        
        input_train_raw = torch.cat(input_train_raw, dim=0)
        target_train_raw = torch.cat(target_train_raw, dim=0)

        train_dataset = TensorDataset(input_train_raw, target_train_raw)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        data_loader_data.append(train_dataloader)
    print(len(data_loader_data))

    def train_data(train_dataloader, scheduler, e_id):

        running_loss = 0.0

        for batch, (input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            pred = task_model(input)
            loss = loss_fn(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 10)

            optimizer.step()
            
            running_loss += loss.item()
        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Lr: {scheduler.get_lr()[0]}, Epoch {e_id+1}/{100}, Loss: {epoch_loss:.7f}")
    
    for e_id in range(epochs):
        train_data(data_loader_data[e_id], scheduler, e_id)

def train_simple_model(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):
    
    freeze = {}

    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, task_infer=None, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        # if args.pull_constraint and 'reduce_sim' in output:
        #     loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        # if task_id > 0:
        #     for (name, param) in model.named_parameters():
        #         if 'head' in name:
        #             key = name.split('.')[0]
        #             param.data = param.data*freeze[key]
            
    #proxy_grad_descent(model, model_old,)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def sample_data(original_model: torch.nn.Module, dataloader_each_class, gm_list, device,
    task_id=-1, args=None):
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Sample: [Task {}]'.format(task_id + 1)

    x_encoded = []

    original_model.eval()
    with torch.no_grad():
        for input, target in metric_logger.log_every(dataloader_each_class, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            #target = target.to(device, non_blocking=True)

            output = original_model(input, task_infer=None)
            x_embed_encode = output['pre_logits']
            x_encoded.append(x_embed_encode)
        x_encoded = torch.cat(x_encoded, dim=0)
        for i in range(10):
            x_encoded_split = x_encoded[(500 * i):(500 * (i+1))]
            gm = GaussianMixture(n_components=1, random_state=0).fit(x_encoded_split.cpu().detach().numpy())
        #gm = GaussianMixture(n_components=1, random_state=0).fit(x_encoded.cpu().detach().numpy())
            gm_list.append(gm)


@torch.no_grad()
def evaluate_task_model(original_model: torch.nn.Module, task_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    
    #criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test task model: [Task {}]'.format(task_id + 1)
    sample_predict_true = 0

    task_model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            #target = target.to(device, non_blocking=True)
            target_logits_raw = torch.Tensor([task_id])
            target_logits = target_logits_raw.expand(input.shape[0], -1).to(device, non_blocking=True)

            # compute output
            if original_model is not None:
                output = original_model(input, task_infer=None)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            logits = task_model(cls_features)

            prob = F.softmax(logits, dim=1)

            task_id_infer = torch.argmax(prob, dim=1)
            task_id_infer = task_id_infer.unsqueeze(1).to(device, non_blocking=True)

            z = torch.eq(task_id_infer, target_logits).to(device, non_blocking=True).sum().item()
            sample_predict_true += z
    
    print("sample_predict_true", sample_predict_true)




@torch.no_grad()
def evaluate_new(model: torch.nn.Module, original_model: torch.nn.Module, task_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    original_model.eval()
    model.eval()
    task_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            if original_model is not None:
                output = original_model(input, task_infer=None)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            logits = task_model(cls_features)

            prob = F.softmax(logits, dim=1)

            task_id_infer = torch.argmax(prob, dim=1)

            #print("task_id_infer", task_id_infer)

            last_logits = model(input, task_infer=task_id_infer)

            loss = criterion(last_logits['logits'], target)

            acc1, acc5 = accuracy(last_logits['logits'], target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now_new(model: torch.nn.Module, original_model: torch.nn.Module, task_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate_new(model=model, original_model=original_model, task_model=task_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)
        
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

        evaluate_task_model(original_model=original_model, task_model=task_model, data_loader=data_loader[i]['val'], device=device,
                            task_id=i,class_mask=class_mask, args=args)

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats
def train_and_evaluate_new(model: torch.nn.Module, original_model: torch.nn.Module, task_model, 
                    criterion, data_loader: Iterable, dataloader_each_class: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, gm_list, device: torch.device, 
                    class_mask=None, args = None,):
    
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):

        #ags-cl
        # if task_id > 0:
        #     freeze = {}
        #     omega = None
        #     for name, param in model.named_parameters():
        #         if 'head' in name:
        #             key = name.split('.')[0]
                    
        #             temp = torch.ones_like(param)
        #             temp = temp.reshape((temp.size(0), omega[prekey].size(0) , -1))
        #             temp[:, omega[prekey] == 0] = 0
        #             temp[omega[key] == 0] = 1
        #             freeze[key] = temp.reshape(param.shape)
        #         else:
        #             continue

        #         prekey = key
        # Transfer previous learned prompt params to the new prompt
        # if args.prompt_pool and args.shared_prompt_pool:
        #     if task_id > 0:
        #         prev_start = (task_id - 1) * args.top_k
        #         prev_end = task_id * args.top_k

        #         cur_start = prev_end
        #         cur_end = (task_id + 1) * args.top_k

        #         if (prev_end > args.size) or (cur_end > args.size):
        #             pass
        #         else:
        #             cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
        #             prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

        #             with torch.no_grad():
        #                 model.e_prompt.grad.zero_()
        #                 model.e_prompt[cur_idx] = model.e_prompt[prev_idx]
        #                 optimizer.param_groups[0]['params'] = model.parameters()
        
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        sample_data(original_model=original_model, dataloader_each_class=dataloader_each_class[task_id]['train_each_class'], gm_list=gm_list, device=device, task_id=task_id, args=args)

        # for epoch in range(args.epochs):
            
        #     train_simple_stat = train_simple_model(model=model, criterion=criterion,
        #                                     data_loader=data_loader[task_id]['train'], optimizer=optimizer,
        #                                     device=device, epoch=epoch, max_norm = args.clip_grad,
        #                                     set_training_mode=True, task_id=task_id, class_mask=class_mask,
        #                                     args=args)
            # if lr_scheduler:
            #     lr_scheduler.step(epoch)
        train_task_model(task_model=task_model, device=device, gm_list=gm_list, epochs=90, task_id=task_id)

        test_stat = evaluate_till_now_new(model=model, original_model=original_model, task_model=task_model, data_loader=data_loader, device=device,
                                        task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args,)
        
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        # log_stats = {**{f'train_{k}': v for k, v in train_simple_stat.items()},
        #     **{f'test_{k}': v for k, v in test_stat.items()},
        #     'epoch': epoch,}

        # if args.output_dir and utils.is_main_process():
        #     with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
        #         f.write(json.dumps(log_stats) + '\n')
        
        #store old head model use ags-cl
        # head_old = deepcopy(model.head)
        # head_old.train()
        # for param in head_old.parameters():
        #     param.requires_grad = False




@torch.no_grad()
def proxy_grad_descent(model: torch.nn.Module, model_old: torch.nn.Module):

    lr = 0.001
    mu = 10
    mask = {}

    for (name,p) in model.named_parameters():
        if 'head' in name:
            name = name.split('.')[:-1]
            name = '.'.join(name)
            mask[name] = torch.zeros(p.shape[0])
    
    with torch.no_grad():
        for (name,module),(_,module_old) in zip(model.named_children(), model_old.named_children()):
            if 'head' in name:
                key = name 
                weight = module.weight
                bias = module.bias

                weight_old = module_old.weight
                bias_old = module_old.bias

                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1,2,3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm**2 + bias**2).pow(1/2)                

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-mask[key])

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1) 
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0