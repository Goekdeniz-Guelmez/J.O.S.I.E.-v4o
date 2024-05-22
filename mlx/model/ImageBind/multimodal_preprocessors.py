import gzip
import html
import io
import math
from functools import lru_cache
from typing import Callable, List, Optional

import ftfy

import numpy as np
import regex as re
import mlx.core as mx
import mlx.nn as nn

from iopath.common.file_io import g_pathmgr
from timm.models.layers import trunc_normal_
