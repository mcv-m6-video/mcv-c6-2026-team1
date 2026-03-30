#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

#Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from util.experiment import build_experiment_name
from dataset.datasets import get_datasets
from model.model_classification import Model

# Evaluation imports
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config json file')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir']
    args.store_dir = os.path.join(config['save_dir'], "splits")
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']

    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    # Auxiliary task
    args.aux_weight = config['aux_weight']

    # Feature aggregation
    args.clip_aggregation = config['clip_aggregation']
    args.attention_heads = config['attention_heads']
    args.transformer_depth = config['transformer_depth']
    args.transformer_dropout = config['transformer_dropout']
    args.transformer_mlp_dim = config['transformer_mlp_dim']

    # Encoder
    args.encoder_arch = config['encoder_arch']
    args.train_last_n_blocks = config.get('train_last_n_blocks', -1)

    # Current experiment
    args.experiment_name = build_experiment_name(args)
    args.run_dir = os.path.join(args.save_dir, args.experiment_name)
    args.ckpt_dir = os.path.join(args.run_dir, "checkpoints")

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = load_json(args.config)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = -float('inf')
        epoch = 0

        print(f'START TRAINING EPOCHS. Experiment {args.experiment_name}')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)
            val_ap_score = evaluate(model, val_data)
            val_map = np.mean(val_ap_score)

            better = False
            if val_map > best_criterion:
                best_criterion = val_map
                better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} Val mAP: {:0.2f}'.format(
                epoch, train_loss, val_loss, val_map * 100))
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_map': float(val_map),
                'is_best': better
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.run_dir, 'history.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'checkpoint_best.pt'))

    print('\nSTART INFERENCE')
    ckpt_path = os.path.join(args.ckpt_dir, 'checkpoint_best.pt')
    map_location = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model.load(torch.load(ckpt_path, map_location=map_location))

    # Evaluation on test split
    ap_score, labels, scores = evaluate(model, test_data)

    # Precision-Recall curve
    for i, class_name in enumerate(classes.keys()):
        precision, recall, _ = precision_recall_curve(labels[:, i], scores[:, i])
        plt.plot(recall, precision, label=f"{class_name} (AP={ap_score[i]*100:.1f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve per Class')
    plt.legend(loc="lower left", fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'pr_curve.png'))
    plt.close()

    # Model stats
    macs, params = model.get_stats()

    # Report results per-class in table
    print('\nPER-CLASS METRICS\n')
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])
    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report average results in table
    print('\nEVALUATION SUMMARY\n')
    headers = ["Metric", "Score"]
    avg_table = [
        ["AP10", f"{np.mean(ap_score[:10])*100:.2f}"],
        ["AP12", f"{np.mean(ap_score)*100:.2f}"]
    ]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    # Report model stats
    print('\nMODEL STATS\n')
    headers = ["Model", "Params", "MACs"]
    model_table = [[args.experiment_name, params, macs]]
    print(tabulate(model_table, headers, tablefmt="grid"))

    print('\nEXECUTION CORRECTLY FINISHED')

if __name__ == '__main__':
    main(get_args())