import os

import torch

from get_latent_action.models.base import LaqBaseModel

from egolam.inference import load_models, encode

class FrozenLamLaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "EgoLAM"
        self.oc, self.model = load_models(os.environ["MODEL_DIR"])
        self.lam = self.model
        self.eval_image_size = 518

    def prepare_model_for_extraction(self):
        self.oc.eval()
        self.model.eval()

    def process_image(self, src_img_path, tgt_img_path, image_size):
        return super().process_image(src_img_path, tgt_img_path, self.eval_image_size)

    def _get_latent_action(self, batch_input):
        tokens = encode(self.oc, self.model, batch_input.permute(0,2,1,3,4)).cpu().numpy()
        return tokens, torch.zeros(len(tokens), 1, dtype=torch.long).numpy()
