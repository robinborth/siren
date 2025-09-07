import numpy as np
import torch
from torch import nn


# See supplement Sec. 1.5 for discussion of factor 30
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(30 * input)


class Siren(nn.Module):
    def __init__(
        self,
        out_features: int = 1,
        in_features: int = 3,
        hidden_features: int = 256,
        num_hidden_layers: int = 3,
    ):
        super().__init__()

        net = []
        net.append(nn.Sequential(nn.Linear(in_features, hidden_features), Sine()))

        for _ in range(num_hidden_layers):
            layer = nn.Sequential(nn.Linear(hidden_features, hidden_features), Sine())
            net.append(layer)

        net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))

        self.net = nn.Sequential(*net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class SirenNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Siren()

    def forward(self, model_input):
        coords = model_input["coords"].clone().detach().requires_grad_(True)
        output = self.net(coords)
        return {"model_in": coords, "model_out": output}
