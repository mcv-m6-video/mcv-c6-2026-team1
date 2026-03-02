import os
import cv2
import argparse
import torch
import json
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from src.detection.runner import build_model
from src.detection.evaluation import evaluate_from_preds

class YOLODatasetAdapter(Dataset):
    """Adapter to read YOLO format images/labels into PyTorch tensors."""
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.lbl_paths = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".txt")])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        boxes, labels = [], []
        if os.path.exists(self.lbl_paths[idx]):
            with open(self.lbl_paths[idx], "r") as f:
                for line in f.readlines():
                    if not line.strip(): continue
                    c, cx, cy, bw, bh = map(float, line.strip().split())
                    # Convert YOLO format back to absolute xyxy for PyTorch
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1) # Shift 0-based YOLO to 1-based FasterRCNN

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, val_loader, val_f_idxs):
    """Runs inference on validation set and evaluates."""
    model.eval()
    preds_by_frame = {}
    i = 0
    with torch.no_grad():
        for images, _ in val_loader:
            f_idxs = val_f_idxs[i : i + len(images)]
            i += len(images)

            preds = model.predict(images)
            for f_id, pred in zip(f_idxs, preds):
                preds_by_frame[f_id] = pred
                
    return evaluate_from_preds(preds_by_frame, test_only=True, preds_dir=None)

def main(args):
    wandb.init(config=args)
    print(f"Training Faster R-CNN | Freeze: {args.freeze_strategy} | LR: {args.lr} | Data: {args.data_dir}")

    # 1. Load Data
    train_ds = YOLODatasetAdapter(
        img_dir=os.path.join(args.data_dir, "images", "train"),
        lbl_dir=os.path.join(args.data_dir, "labels", "train")
    )
    val_ds = YOLODatasetAdapter(
        img_dir=os.path.join(args.data_dir, "images", "val"),
        lbl_dir=os.path.join(args.data_dir, "labels", "val")
    )
    with open(os.path.join(args.data_dir, "splits.json"), "r") as f:
        val_f_idxs = json.load(f)["val"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 2. Setup Model
    model = build_model("faster_rcnn", device=args.device)
    device = model.device
    trainable_params = model.get_trainable_params(freeze_strategy=args.freeze_strategy)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Train Loop with Early Stopping
    best_ap50 = 0.0
    patience_counter = 0
    patience_limit = 5
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for images, targets in train_loader:
            images = [to_tensor(img).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        avg_loss = epoch_loss / max(1, len(train_loader))
        scheduler.step()
        
        # Evaluate
        val_ap50 = evaluate(model, val_loader, val_f_idxs)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val AP50: {val_ap50:.4f}")
        wandb.log({"train_loss": avg_loss, "val_AP50": val_ap50, "epoch": epoch + 1})
        
        # Early Stopping Logic
        if val_ap50 > best_ap50:
            best_ap50 = val_ap50
            patience_counter = 0
            torch.save(model.state_dict(), f"best_faster_rcnn_f{args.freeze_strategy}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}. Best AP50: {best_ap50:.4f}")
                break

    print(f"Finished training Faster R-CNN | Freeze: {args.freeze_strategy} | LR: {args.lr} | Data: {args.data_dir}")

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
