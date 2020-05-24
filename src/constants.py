"""Simple constants and typedefs used throughout the project."""

from typing import Union

import torchvision.transforms as transforms


Transforms = Union[
    transforms.Compose,
    transforms.CenterCrop,
    transforms.ColorJitter,
    transforms.FiveCrop,
    transforms.Grayscale,
    transforms.Pad,
    transforms.RandomAffine,
    transforms.RandomApply,
    transforms.RandomChoice,
    transforms.RandomCrop,
    transforms.RandomGrayscale,
    transforms.RandomHorizontalFlip,
    transforms.RandomOrder,
    transforms.RandomPerspective,
    transforms.RandomResizedCrop,
    transforms.RandomRotation,
    transforms.RandomSizedCrop,
    transforms.RandomVerticalFlip,
    transforms.Resize,
    transforms.Scale,
    transforms.TenCrop,
    transforms.LinearTransformation,
    transforms.Normalize,
    transforms.RandomErasing,
    transforms.ToPILImage,
    transforms.ToTensor,
    transforms.Lambda,
]