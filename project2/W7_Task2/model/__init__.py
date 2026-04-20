from .model_spotting import Model as BaselineModel
from .model_spotting_x3d import Model as X3DModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model type: BASELINE")
        return BaselineModel(args)
    elif args.model_type == "x3d":
        print("Using model type: X3D")
        return X3DModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]