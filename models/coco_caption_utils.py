import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel


def initialize_clip(config, num_patches=240):
    clip_preprocessor = CLIPProcessor.from_pretrained(config['clip_name'])
    clip_model = CLIPModel.from_pretrained(config['clip_name'])
    return clip_model, clip_preprocessor