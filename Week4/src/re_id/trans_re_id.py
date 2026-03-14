import numpy as np

class TransReID:
    """
    Interface for the damo-cv/TransReID repository.
    Requires cloning: git clone https://github.com/damo-cv/TransReID.git

    TODO: ADAPT/TRAIN, ETC. (ORIOL)
    """
    def __init__(self, config_path, model_weights_path, device='cuda'):
        self.device = device
        # The exact import depends on the TransReID repository structure.
        # Typically, you build the model and load the state dict:
        # from model import make_model 
        # self.model = make_model(...)
        # self.model.load_param(model_weights_path)
        # self.model.eval()
        # self.model.to(self.device)
        print(f"TransReID model active on {self.device}.")
        
    def extract_features(self, image_crops):
        """
        Processes a batch of cropped vehicle images and returns their embeddings.
        """
        # Example execution block:
        # with torch.no_grad():
        #     features = self.model(image_crops)
        #     return features.cpu().numpy()
        
        # Returns a mock feature matrix for structural demonstration
        return np.random.rand(len(image_crops), 768)