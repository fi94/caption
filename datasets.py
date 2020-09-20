import torch
from torch.utils.data import Dataset
import torch.utils.data
import torchvision.transforms as transforms
import h5py
import json
import os
import torch.utils.data
from tqdm import tqdm
from models import Encoder


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
 
        self.split = split
        
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, '{}_IMAGE.hdf5'.format(split)), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = 2

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder,  '{}_CAPTIONS.json'.format(split)), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, '{}_CAPLENS.json'.format(split)), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        
        

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        #print(img.size())
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
    
    
if __name__ == '__main__':
    data_folder = './process_data'
    workers = 0
    batch_size = 4
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'val', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    encoder = Encoder()
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        print(imgs.shape)
        #imgs = encoder(imgs)
        #print(caps)