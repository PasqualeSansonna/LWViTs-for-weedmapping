import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F


class PairRandomCrop:
    """Crop the given PIL.Image at a random location.
    ** This is a MODIFIED version **, which supports identical random crop for
    both image and target map in Semantic Segmentation.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    image_crop_position = {}

    def __init__(self, size, padding=0):
        import random
        import os
        import numbers

        from PIL import ImageOps

        self.os = os
        self.random = random
        self.ImageOps = ImageOps

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = self.ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        pid = self.os.getpid()
        if pid in self.image_crop_position:
            x1, y1 = self.image_crop_position.pop(pid)
        else:
            x1 = self.random.randint(0, w - tw)
            y1 = self.random.randint(0, h - th)
            self.image_crop_position[pid] = (x1, y1)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class ToLong:
    def __call__(self, x):
        return x.long()


class SegOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        y = torch.moveaxis(one_hot(x, self.num_classes), 2, 0)
        return y


class FixValue:
    def __init__(self, source, target):
        self.s = source
        self.t = target

    def __call__(self, x):
        x[x == self.s] = self.t
        return x


class Denormalize(torch.nn.Module):
    """Denormalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return tensor.mul(std.view(-1, 1, 1)).add(mean.view(-1, 1, 1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"