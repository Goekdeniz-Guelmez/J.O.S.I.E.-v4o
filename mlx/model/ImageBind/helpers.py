import math

import einops
import numpy as np
import mlx.nn as nn
import mlx.core as mx


class LearnableLogitScaling(nn.Module):
    """Skalierung der Logits mit einem lernbaren oder festen Skalierungsfaktor."""
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ):
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable

        log_logit_scale = mx.ones([]) * mx.log(self.logit_scale_init)

        # Wenn skalierbar, wird log_logit_scale als Parameter registriert
        if self.learnable:
            self.log_logit_scale = log_logit_scale

    def __call__(self, x):
        # Skaliert den Eingabetensor x unter Berücksichtigung des maximalen Skalenwertes
        return mx.clip(a_min=self.log_logit_scale.exp(), a_max=self.max_logit_scale) * x


class EinOpsRearrange(nn.Module):
    """Modul zur Neuanordnung von Tensoren mit EinOps."""
    def __init__(
        self,
        rearrangee_expr: str,
        **kwargs
    ):
        super().__init__()
        self.rearrange_expr = rearrangee_expr
        self.kwarg = kwargs

    def __call__(self, x):
        assert isinstance(x, mx.array)
        # Rearrangiere den Tensor x basierend auf dem EinOps-Ausdruck
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


class SelectElement(nn.Module):
    """Modul zur Auswahl eines bestimmten Elements entlang einer Dimension."""
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, x):
        assert x.ndim >= 3
        # Wählt das Element mit dem gegebenen Index entlang der zweiten Dimension
        return x[:, self.index, ...]


class SelectEOSAndProject(nn.Module):
    """Text-Pooling-Modul basierend auf OpenCLIP."""
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def __call__(self, x, seq_len):
        assert x.ndim == 3
        # x hat die Form B x L x D
        # Wählt Merkmale aus dem EOT-Embedding (Ende-der-Sequenz-Token)
        x = x[mx.arange(x.shape[0]), seq_len]
        # Projektionsschritt
        x = self.proj(x)
        return x
