import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .criterion import Custom_Loss  # noqa
from .datasets import CocoDataset2


def get_device(device: str):
    return torch.device(device)


def get_loader(config: dict, phase: str):
    dataset_config = config["data"]
    dataset_config['mode'] = phase

    loader_config = config["loader"][phase]

    dataset = CocoDataset2(dataset_config.copy())
    loader = DataLoader(dataset, **loader_config)

    return loader


def get_criterion(config: dict):
    loss_config = config["criterion"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get(
        "params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls()
        else:
            raise NotImplementedError

    return criterion


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")
    optimizer = optim.__getattribute__(optimizer_name)([
        {'params': [*model.parameters()][20:24],
         'lr': optimizer_config["params"]["lr"] / 4},
        {'params': [*model.parameters()][24:],
         'lr': optimizer_config["params"]["lr"]}])

    return optimizer


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])
