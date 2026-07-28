import os

import torch
from torch import nn
from collections import OrderedDict
from merlin.models.build import MerlinArchitecture,MerlinArchitecture_classify,MerlinArchitecture_modified
from merlin.utils import download_file
import ipdb

class Merlin(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super(Merlin, self).__init__()
        self.ImageEmbedding = ImageEmbedding
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        self._download_checkpoint()
        model = MerlinArchitecture(ImageEmbedding=self.ImageEmbedding)
        # model.load_state_dict(
        #     torch.load(os.path.join(self.local_dir, self.checkpoint_name)),
        #     strict=False
        # )
        full_state_dict = torch.load(os.path.join(self.local_dir, self.checkpoint_name))
        filtered_state_dict = OrderedDict()
        print("Filtering state_dict to keep only 'encode_image' weights...")
        for key, value in full_state_dict.items():
            if key.startswith("encode_image.") and "encode_image.i3_resnet.classifier" not in key and "encode_image.i3_resnet.contrastive_head" not in key:
                filtered_state_dict[key] = value
        print("Loading filtered state_dict into the model...")
        model.load_state_dict(
            filtered_state_dict,
            strict=True
        )
        return model

    """ 
    Download the Merlin weights from the Hugging Face Hub
    """

    def _download_checkpoint(self):
        download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input):
        return self.model(*input)
    
class Merlin_modified(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super(Merlin_modified, self).__init__()
        self.ImageEmbedding = ImageEmbedding
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        self._download_checkpoint()
        model = MerlinArchitecture_modified(ImageEmbedding=self.ImageEmbedding)
        # ipdb.set_trace()
        # full_state_dict = torch.load(os.path.join(self.local_dir, self.checkpoint_name))
        # filtered_state_dict = OrderedDict()
        # print("Filtering state_dict to keep only 'encode_image' weights...")
        # for key, value in full_state_dict.items():
        #     if key.startswith("encode_image.") and "encode_image.i3_resnet.classifier" not in key and "encode_image.i3_resnet.contrastive_head" not in key:
        #         filtered_state_dict[key] = value
        # print("Loading filtered state_dict into the model...")
        # model.load_state_dict(
        #     filtered_state_dict,
        #     strict=True
        # )
        # model.load_state_dict(
        #     torch.load(os.path.join(self.local_dir, self.checkpoint_name)),
        #     strict=False
        # )
        return model

    """ 
    Download the Merlin weights from the Hugging Face Hub
    """

    def _download_checkpoint(self):
        download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input):
        return self.model(*input)


class Merlin_classify(nn.Module):
    def __init__(self, ImageEmbedding: bool = False):
        super(Merlin_classify, self).__init__()
        self.ImageEmbedding = ImageEmbedding
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "stanfordmimi/Merlin"
        self.model = self._load_model()

    """
    Load the Merlin model with the initialized weights
    """

    def _load_model(self):
        self._download_checkpoint()
        model = MerlinArchitecture_classify(ImageEmbedding=self.ImageEmbedding)
        model.load_state_dict(
            torch.load(os.path.join(self.local_dir, self.checkpoint_name)),
            strict=False
        )
        return model

    """ 
    Download the Merlin weights from the Hugging Face Hub
    """

    def _download_checkpoint(self):
        download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input):
        # ipdb.set_trace()
        return self.model(*input)
