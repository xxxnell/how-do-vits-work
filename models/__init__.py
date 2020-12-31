import os
from pathlib import Path

import torch


def get_model(name, num_classes, tiny=False, **kwargs):
    raise NotImplemented


def save(model, dataset_name, uid, optimizer=None):
    checkpoint_path = os.path.join("models_checkpoints", dataset_name, model.name)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(checkpoint_path, "%s_%s_%s.pth.tar" % (dataset_name, model.name, uid))
    save_obj = {
        "name": model.name,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        save_obj["optimizer"] = optimizer.state_dict()

    torch.save(save_obj, save_path)


def load(model, dataset_name, uid, optimizer=None):
    checkpoint_path = os.path.join("models_checkpoints", dataset_name, model.name)
    save_path = os.path.join(checkpoint_path, "%s_%s_%s.pth.tar" % (dataset_name, model.name, uid))
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def stats(model):
    model.eval()
    xs = torch.randn([3, 3, 32, 32])
    ys = model(xs)
    print("model: %s (%.1fM), output: %s" % (model.name, count_parameters(model) / 1e6, ys.size()))


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

