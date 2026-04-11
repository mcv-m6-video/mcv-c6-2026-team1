#!/usr/bin/env python3
"""
File containing the main training script.
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
from torch.utils.data import DataLoader, RandomSampler
from tabulate import tabulate

#Local imports
from util.io import load_json, store_json, save_video
from util.eval_spotting import evaluate, generate_qualitative_results
from dataset.datasets import get_datasets
from model.model_spotting import Model

BACKGROUND_LABEL = "BACKGROUND"

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
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
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.qualitatives = config['qualitatives']
    args.device = config['device']
    args.num_workers = config['num_workers']
    args.early_stopping_patience = config["early_stopping_patience"]

    # Bets model criterion
    args.use_ap10 = config["use_ap10"]

    # Run directory
    args.run_dir = os.path.join(args.save_dir, args.model)

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

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model data
    os.makedirs(args.run_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, val_video_data, test_video_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly!')
        print('Generating dataset statistics...')
        stats = {}
        for split_name, split in [("train", train_data), ("val", val_data), ("test", test_video_data)]:
            counts = {name: 0 for name in classes.keys()}
            counts[BACKGROUND_LABEL] = 0
            total_frames = 0
            for video in split._games:
                labels_file = load_json(os.path.join(split._labels_dir, video['video'] + '/Labels-ball.json'))['annotations']
                num_frames = int(video['num_frames'])
                action_frames = 0
                for event in labels_file:
                    counts[event['label']] += 1
                    action_frames += 1
                        
                counts[BACKGROUND_LABEL] += num_frames - action_frames
                total_frames += num_frames
            
            stats[split_name] = {
                "total_frames": total_frames,
                "counts": counts
            }
            
        # Save the dictionary to JSON
        stats_path = os.path.join(args.save_dir, 'dataset_statistics.json')
        store_json(stats_path, stats, pretty=True)
        print(f"Dataset statistics saved to {stats_path}")

        sys.exit('Re-run changing "mode" to "load" in the config JSON for training/inference.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    # Dataloaders
    clips_per_epoch = args.epoch_num_frames // args.clip_len
    train_sampler = RandomSampler(train_data, replacement=True, num_samples=clips_per_epoch)
    train_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size, 
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None)
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None)
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
        best_criterion = -float('inf') if args.use_ap10 else float("inf")
        epoch = 0
        patience_counter = 0

        print(f'START TRAINING EPOCHS ({args.model})')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)
            map_score, ap_score = evaluate(model, val_video_data)
            val_ap10 = np.mean(ap_score[:10]) # Leave out free kick and goal
            criterion_value = val_ap10 if args.use_ap10 else val_loss

            better = False
            if criterion_value < best_criterion:
                best_criterion = criterion_value
                better = True
                patience_counter = 0

            else:
                patience_counter += 1
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} mAP10: {:0.5f}'.format(
                epoch, train_loss, val_loss, val_ap10))
            if better:
                print('New best epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss, 'mAP10': val_ap10
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.run_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(args.run_dir, 'checkpoint_best.pt') )

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print('\nSTART INFERENCE')
    model.load(torch.load(os.path.join(args.run_dir, 'checkpoint_best.pt')))

    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_video_data)

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
        ["AP12", f"{np.mean(ap_score)*100:.2f}"],
        ["SoccerNet mAP", f"{map_score*100:.2f}"]
    ]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    # Report model stats
    print('\nMODEL STATS\n')
    headers = ["Model", "Params", "MACs"]
    model_table = [[args.model, params, macs]]
    print(tabulate(model_table, headers, tablefmt="grid"))

    if args.qualitatives:
        print('\nGenerating qualitative results...')
        qualitative_results = generate_qualitative_results(model, test_video_data, BACKGROUND_LABEL, args.labels_dir)
        qualitative_dir = os.path.join(args.run_dir, 'qualitative_results')
        os.makedirs(qualitative_dir, exist_ok=True)
        
        json_dump = []
        for idx, result in enumerate(qualitative_results):
            video_path = os.path.join(qualitative_dir, f"sample_{idx}.mp4")
            save_video(video_path, result["frames"])
            json_dump.append({
                "path": video_path,
                "video": result["video"],
                "start": result["start"],
                "end": result["end"],
            })
            
        json_path = os.path.join(qualitative_dir, 'qualitative.json')
        store_json(json_path, json_dump, pretty=True)
    
    print('\nEXECUTION CORRECTLY FINISHED')


if __name__ == '__main__':
    main(get_args())