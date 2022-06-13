# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils


def cal_correct(text_output, target):
    # cal_total_correct
    text_output = text_output.softmax(-1).max(-1).indices
    equal = text_output == target

    all_correct = equal.sum()
    all_num = text_output.numel()

    instanse_correct = equal.sum(-1).eq(6)
    instanse_num = text_output.shape[0]

    return all_correct/all_num, instanse_correct/instanse_num


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    tb_logger=None,  cl_start_ep = 100,start_idx=0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ins_acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    torch.cuda.synchronize()    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        losses = criterion(outputs, targets, text_coef = 1.0 if epoch > cl_start_ep else 0.,metric_logger = metric_logger)

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        total_acc,  ins_acc = cal_correct(outputs[-1], target=targets['text'])

        torch.cuda.synchronize()
        metric_logger.update(loss_total=loss_value)
        metric_logger.update(total_acc=total_acc)
        metric_logger.update(ins_acc=ins_acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if tb_logger is not None and utils.get_rank() == 0 and start_idx % 50 == 0:
            for k, meter in metric_logger.meters.items():
                tb_logger.add_scalar('train/{}_avg'.format(k), meter.global_avg, start_idx)
                tb_logger.add_scalar('train/{}_val'.format(k), meter.value, start_idx)
        start_idx += 1



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_boxes', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('total_acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('ins_acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        losses = criterion(outputs, targets, metric_logger = metric_logger)
        loss_value = losses.item()
        total_acc,  ins_acc = cal_correct(outputs[-1], target=targets['text'])

        torch.cuda.synchronize()
        metric_logger.update(loss_total=loss_value)
        metric_logger.update(total_acc=total_acc)
        metric_logger.update(ins_acc=ins_acc)
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
