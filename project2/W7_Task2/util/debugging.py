import numpy as np
from torch.utils.data import DataLoader


def debug_predictions(model, dataset, classes, num_samples=5):
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    class_names = ["BACKGROUND"] + list(classes.keys())

    print("\nDEBUGGING PREDICTIONS VS GT\n")

    for idx, batch in enumerate(loader):
        if idx >= num_samples:
            break

        frame = batch["frame"]
        gt_labels = batch["label"][0]
        gt_timestamps = batch["timestamp"][0]

        pred = model.predict(frame)
        probs = pred["probs"][0]          # [Q, C+1]
        timestamps = pred["timestamps"][0]  # [Q, 1]

        print(f"\n--- SAMPLE {idx} ---")

        # GT
        if len(gt_labels) == 0:
            print("GT: no events")
        else:
            print("GT events:")
            for j in range(len(gt_labels)):
                cls_id = gt_labels[j].item()
                t = gt_timestamps[j].item()
                print(f"  - class={class_names[cls_id]} ({cls_id}), time={t:.3f}")

        # Predictions
        print("Predicted queries:")
        for q in range(probs.shape[0]):
            pred_class = np.argmax(probs[q])
            pred_score = probs[q, pred_class]
            pred_time = timestamps[q, 0]

            top3 = np.argsort(probs[q])[::-1][:3]
            top3_str = ", ".join(
                [f"{class_names[c]}:{probs[q, c]:.3f}" for c in top3]
            )

            print(
                f"  q={q:02d} | pred={class_names[pred_class]} ({pred_class}) "
                f"| score={pred_score:.3f} | time={pred_time:.3f} "
                f"| top3=[{top3_str}]"
            )