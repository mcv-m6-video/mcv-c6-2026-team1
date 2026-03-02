import argparse
import wandb
from src.detection.runner import build_model

def main(args):
    wandb.init(config=args, name=f"yolo_freeze{args.freeze_strategy}_lr{args.lr}")
    
    # Initialize wrapper
    model = build_model("yolo", device=args.device)
    print(f"Training YOLO | Freeze: {args.freeze_strategy} | LR: {args.lr} | Data: {args.data_dir}")
    
    # Callback to inject metrics into W&B
    def on_fit_epoch_end(trainer):
        val_ap50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        train_loss = trainer.loss_items.sum().item() if hasattr(trainer, 'loss_items') and trainer.loss_items is not None else 0.0
        epoch = trainer.epoch + 1
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val AP50: {val_ap50:.4f}")
        wandb.log({
            "val_AP50": val_ap50,
            "train_loss": train_loss,
            "epoch": epoch
        })

    # Attach the callback before training starts
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Train natively using Ultralytics engine
    model.train(args)

    print(f"Finished training YOLO | Freeze: {args.freeze_strategy} | LR: {args.lr} | Data: {args.data_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--freeze_strategy", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)