import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import PretrainedConfig

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel

import numpy as np

# moved into `dataset.py`
class StrctDataset(Dataset):
    """Dataset wrapping tensors

    Each sample will be retrieved by indexing tensors along the first dimension
    Later the dataset is sent into a sampler then a dataloader, and transfer to `input` in training phase

    Arguments:
        *tensors (*torch.Tensor): tensors having the same size of the first dimension. 
            For example (all_input_ids, all_input_masks, all_segment_ids, all_feature_index)
        page_id (list): the corresponding page_ids of the input features.
        token_to_tag (torch.Tensor): the mapping from each token to its corresponding tag id
    """

    def __init__(self, *tensors, page_ids=None, token_to_tag=None):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Invalid input tensors with different size in the 1st dimension, expected same size."
        self.tensors = tensors
        self.page_ids = page_ids
        self.token_to_tag = token_to_tag

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) # return the indexed tensors if no features from other modalities is required

    def __len__(self):
        return len(self.tensors[0])


# Lists of strs storing the name of backbones and methods, corresponding to different (total) model TODO
__BACKBONES__ = ["baseline", "bart"] # backbones pretrained models supported in the model, baseline is Bart
__METHODS__ = ["baseline", "text_html"] # features-using methods supported in the model, baseline is textual and html features

def to_model(backbone, method, config):
    if backbone == "bart" and method == "text_html":
        model_config = BartConfig(config)
        return Bart(model_config)
    else:
        raise NotImplementedError("A combination of backbone and method unrecognizable")


class BaseModelConfig(PretrainedConfig):
    """The configuration base class to store the basic configurations of all models

    Arguments:
        backbone (str): the name of the backbone models, see __BACKBONES__
        method (str): the name of the methods using different modalities or other features, see __METHODS__
        kwargs (dict): correspond to the `PretrainedConfig` usage
    """
    def __self__(self, backbone, method, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone if backbone in __BACKBONES__ else "baseline"
        self.method = method if method in __METHODS__ else "baseline"

class Backbone(nn.Module):
    """
    The wrapper class for backbone pretrained model
    """
    def __init__(self, config):
        pass

class QuesGenModel(nn.Module):
    """
    The model class wrapper for all backbones and methods
    """
    pass

class Bart(BartPretrainedModel):
    """
    """
    def __init__(self, config):
        super(Bart, self).__init__(config) # diff super() <-> super(Bart, self) ?
        self.

    def forward(
        self,
        input 
        )