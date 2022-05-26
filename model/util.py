import torch
import numpy as np

from typing import Optional


def xyxy2mask(x1, y1, x2, y2, H, W):
    x1 = float2int(x1)
    x2 = float2int(x2)
    y1 = float2int(y1)
    y2 = float2int(y2)
    mask = np.zeros((H, W))
    mask[y1:y2, x1:x2] = 1.
    return mask


def xyxy2mask_ts(x1, y1, x2, y2, H, W):
    x1 = max(min(x1.long(), W), 0)
    y1 = max(min(y1.long(), W), 0)
    x2 = max(min(x2.long(), W), 0)
    y2 = max(min(y2.long(), W), 0)
    mask = torch.zeros((H, W))
    mask[y1:y2, x1:x2] = 1.
    return mask


def xyxy2xyhw(box):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = abs(x2-x1)
    h = abs(y2-y1)
    return [xc, yc, h, w]


def rchw2xyxy(rchw, H, W):
    xyxy = np.zeros_like(rchw)
    y = rchw[..., 0]
    x = rchw[..., 1]
    h = np.abs(rchw[..., 2])
    w = np.abs(rchw[..., 3])
    xyxy[..., 0] = np.maximum(0, x-w//2)  # x1
    xyxy[..., 1] = np.maximum(0, y-h//2)  # y1
    xyxy[..., 2] = np.minimum(W, xyxy[..., 0]+w)  # x2
    xyxy[..., 3] = np.minimum(H, xyxy[..., 1]+h)  # y2
    return xyxy


def float2int(val):
    return int(round(val))

def deparametric(v, v_a, ratio=1.):
    return np.exp(v) * v_a / ratio


def deparametric_ts(v, v_a, ratio=1.):
    return torch.exp(v) * v_a / ratio


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
