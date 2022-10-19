import os
import json

import torch
from torch.utils.data import Dataset

import bs4
from bs4 import BeautifulSoup as bs

"""
the train/eval json file (already processed) is in the nested structure like:
version, data
         -> domain (dir: sport, auto, ...), websites
                                            -> qas                                 , page id
                                               -> question, id, answer
                                                  -> text, element id, answer start 
"""

class AutoDataset(Dataset):
    pass

class TextDataset(Dataset):
    pass