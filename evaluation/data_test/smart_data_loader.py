import numpy as np
import torch
from torch.utils import data
# from skimage.measure import label
# from scipy import ndimage
# from scipy.ndimage.morphology import binary_dilation
# from scipy.ndimage import distance_transform_edt
import cv2
from PIL import Image
from data_test.data_aug import transformation
import os

import pdb


class Data(data.Dataset):
    def __init__(self, image_dir, gt_dir, data_aug=None, aug_mode=None, unseen=False, mode='train'):
        self.image_dir = image_dir
        self.gt_dir    = gt_dir
        self.data_aug  = data_aug
        self.aug_mode  = aug_mode

        self.image_path = os.listdir(image_dir)
        self.gt_path = os.listdir(gt_dir)    
        num_samples = len(self.image_path)
        if mode == 'train':
            self.image_path = self.image_path[:int((num_samples/3)*2)]
            self.gt_path = self.gt_path[:int((num_samples/3)*2)]
            print('Training data:{}'.format(int((num_samples/3)*2)))
        if mode == 'test':
            self.image_path = self.image_path[:]
            self.gt_path = self.gt_path[:]
            print('Testing data:{}'.format(int((num_samples))))
        else:
            num_samples = len(self.image_path)
            self.image_path = self.image_path[int((num_samples/3)*2):]
            self.gt_path = self.gt_path[int((num_samples/3)*2):]
            print('Validation data:{}'.format(int((num_samples/3))))
        

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.image_dir, self.image_path[index]))

        labels = cv2.imread(os.path.join(self.gt_dir, self.gt_path[index]), 0)
        labels = labels / 255.
        labels = labels.astype(np.uint8)

        if self.data_aug:
            img, labels = transformation(img, labels, self.aug_mode)
        img = img / 255.

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        labels = torch.from_numpy(np.array([labels])).float()
        return img, labels
