import torch
import os
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image




def load_data_list(data_dir):
     path = os.path.join(data_dir, '', '*')
     file_list = glob(path)
     return file_list

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def return_loader(crop_size, image_size, batch_size, mode='train'):
    """Return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    shuffle = False

    if mode == 'train':
        shuffle = True

    #Path to image folders of expression classes for both image and landmark heatmap
    traindir_img='/data/train_face_data/image/'
    traindir_heatmap='/data/train_face_data/landmark/'


    data_loader =DataLoader(
             dataset=ConcatDataset(
                 ImageFolder(traindir_img,transform),
                 ImageFolder(traindir_heatmap,transform)
             ),
             batch_size=batch_size, shuffle=shuffle)
    return data_loader
