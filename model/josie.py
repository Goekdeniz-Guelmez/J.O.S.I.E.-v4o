# The main J.O.S.I.E.v4o model file. It's heavely barrowed from NeXT-GPT, big thanks.
import os
from typing import List

from ImageBind.imagebind import *
from ImageBind.imagebind import data

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


class JOSIE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args['max_length']

        self.stage = args['stage']

        print(f"Initialized model with max context length: {self.max_length}")

        imagebind_encoder_path = os.path.join(self.args['imagebind_encoder_path'])
