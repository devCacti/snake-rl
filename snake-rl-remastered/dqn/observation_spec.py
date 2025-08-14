import torch


class ObservationSpec:
    def __init__(self, shape, dtype=torch.float32, low=-1.0, high=1.0):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high
