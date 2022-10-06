import torch
import torch.nn as nn
from torch.utils.data import Dataset

class StrucDataset(Dataset):
    """Dataset wrapping tensors, baseline method, with only text features and html tag features

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