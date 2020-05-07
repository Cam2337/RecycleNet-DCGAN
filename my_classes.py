# Relevant imports
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


## Following the Pytorch tutorial: 
## https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""
Example call:

data_set = TrashDataset(csv_file = 'trash.csv',
						root_dir='data/trash/)

transformed_dataset = TrashDataset(csv_file = 'trash.csv',
                                           root_dir='data/trash/',
                                           transform=transforms.Compose([
                                               RandomHorizontalFlip(0.5),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

input_data = ConcatDataset([dataset, transformed_dataset])

dataloader = DataLoader(input_data, batch_size=4,
                        shuffle=True, num_workers=4)


"""

class TrashDataset(Dataset):
    """Trash dataset."""

    def __init__(self, file, root_dir, transform=None):
        """
        Args:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # The image file names and classes
        self.trash_frame = pd.read_csv('output_list.txt', sep = " ")
        # Where the file is located
        self.root_dir = root_dir
        # Transforms, if supplied by the user
        self.transform = transform

    def __len__(self):
        return len(self.trash_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find the image name at the given index
        img_name = os.path.join(self.root_dir,
                                self.trash_frame.iloc[idx, 0])
        
        image = io.imread(img_name)
        trash = self.trash_frame.iloc[idx, 1:]
        trash = np.array([trash])
        trash = trash.astype('float').reshape(-1, 2)
        sample = {'image': image, 'trash': trash}

        if self.transform:
            sample = self.transform(sample)

        return sample



