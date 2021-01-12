import torch
import torch.nn as nn


class Ensemble(nn.Module):

    def __init__(self, models, name=None):
        super(Ensemble, self).__init__()

        if name is not None:
            self.name = name
        else:
            self.name = "%s_ensemble" % models[0].name

        self.models = nn.ModuleList(models)

    def forward(self, x):
        xs = torch.stack([model(x) for model in self.models])
        xs = xs - torch.logsumexp(xs, dim=-1, keepdim=True)
        x = torch.logsumexp(xs, dim=0)

        return x

