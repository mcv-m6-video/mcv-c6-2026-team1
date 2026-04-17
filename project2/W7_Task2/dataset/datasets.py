"""
File containing the function to load all the frame datasets (and the DETR collate function).
"""

#Standard imports
import os
import torch

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset

#Constants
DEFAULT_STRIDE = 2      # Sampling stride (if greater than 1, frames are skipped) / Effectively reduces FPS
DEFAULT_OVERLAP = 0.9   # Temporal overlap between sampled clips (for traiing and validation only)

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if "stride" in args else DEFAULT_STRIDE
    overlap = args.overlap if "overlap" in args else DEFAULT_OVERLAP

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, **dataset_kwargs)
    train_data.print_info()

    dataset_kwargs['overlap'] = 0

    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, **dataset_kwargs)
    val_data.print_info()     

    # Not needed for testing
    dataset_kwargs.pop('label_radius', None)

    # Need raw video data to compute mAP10 for validation and testing
    val_video_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, **dataset_kwargs)
    val_video_data.print_info()

    test_video_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.clip_len, **dataset_kwargs)
    test_video_data.print_info()
        
    return classes, train_data, val_data, val_video_data, test_video_data

def detr_collate_fn(batch):
    frames = torch.stack([item['frame'] for item in batch])
    contains_event = torch.tensor([item['contains_event'] for item in batch])
    labels = [item['label'] for item in batch]
    timestamps = [item['timestamp'] for item in batch]
    return {'frame': frames, 'contains_event': contains_event, 'label': labels, 'timestamp': timestamps}