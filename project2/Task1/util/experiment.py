import hashlib
import json
import os


def build_experiment_name(args):
    tracked = {
        "encoder_arch": args.encoder_arch,
        "clip_aggregation": args.clip_aggregation,
        "clip_len": args.clip_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "aux_weight": args.aux_weight,
        "seed": args.seed,
    }

    if args.clip_aggregation.lower() == "transformer":
        tracked.update({
            "attention_heads": args.attention_heads,
            "transformer_depth": args.transformer_depth,
            "transformer_dropout": args.transformer_dropout,
            "transformer_mlp_dim": args.transformer_mlp_dim,
        })

    short_hash = hashlib.md5(
        json.dumps(tracked, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    readable_name = (
        f"{args.encoder_arch}_"
        f"{args.clip_aggregation}_"
        f"lr{args.learning_rate}_"
        f"aux{args.aux_weight}_"
        f"seed{args.seed}_"
        f"train_n_layers{args.train_last_n_blocks}_"
    )

    return f"{readable_name}_{short_hash}"