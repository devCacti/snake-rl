import torch


def to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)
