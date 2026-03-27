import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch

EXTERNAL_PATH = Path(__file__).resolve().parents[3] / "external"
TRANS_RE_ID_ROOT = EXTERNAL_PATH / "TransReID"
sys.path.insert(0, str(TRANS_RE_ID_ROOT))
sys.path.insert(0, str(EXTERNAL_PATH))

from config import cfg as transreid_cfg
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
from utils.logger import setup_logger

from src.re_id.build_reid_dataset import make_custom_dataloaders, build_reid_dataset


DATASET_PATH = "./data/reid_custom_dataset"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train TransReID on custom vehicle ReID dataset")
    parser.add_argument("--config_file", type=str, required=True, help="Path to TransReID yaml config")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to built dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save logs/checkpoints")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Additional config overrides, e.g. SOLVER.MAX_EPOCHS 60",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(DATASET_PATH):
        build_reid_dataset(
            sequence_ids=[1, 4],
            output_dir=DATASET_PATH,
            max_crops_per_id_per_camera=20,
            min_box_area=500,
            train_ratio=0.8,
        )

    cfg = transreid_cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["OUTPUT_DIR", args.output_dir])

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logger = setup_logger("transreid_custom", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Training custom TransReID")
    logger.info(f"Config file : {args.config_file}")
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Output dir  : {cfg.OUTPUT_DIR}")
    logger.info(f"Running with config:\n{cfg}")

    set_seed(cfg.SOLVER.SEED)

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = \
        make_custom_dataloaders(cfg, args.dataset_root)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
    )

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        local_rank=0,
    )


if __name__ == "__main__":
    main()