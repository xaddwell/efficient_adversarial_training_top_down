import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random
import torch.utils.data as data
import pickle
from tqdm import tqdm


to_tensor = transforms.ToTensor()

class train_imageNet_datasets(Dataset):
    def __init__(self, filename, shuffle=False):
        print("=====>>>load train_imageNet_datasets from {}".format(filename))
        ori_dir = filename + '/ori/'
        class_list=os.listdir(ori_dir)
        self.class_file=[]
        for name in class_list:
            temp_file = ori_dir +name
            label = name.split('.')[0].split('_')[-1]
            advs_dir = filename + '/advs/' + name
            self.class_file.append((temp_file,advs_dir,label))
        if shuffle:
            random.shuffle(self.class_file)

    def __getitem__(self, idx):
        ori,advs,label =self.class_file[idx]
        ori = to_tensor(Image.open(ori))
        advs = to_tensor(Image.open(advs))
        label = torch.tensor(int(label))
        return ori,advs,label

    def __len__(self):
        return len(self.class_file)

class compare_imageNet_datasets(Dataset):
    def __init__(self, filename, shuffle=True,transform=None):
        print("=====>>>load compare_imageNet_datasets from {}".format(filename))
        ori_dir = filename + '/ori/'
        class_list=os.listdir(ori_dir)
        self.transform = transform
        self.class_file=[]
        for name in class_list:
            temp_file = ori_dir +name
            label = name.split('.')[0].split('_')[-1]
            advs_dir = filename + '/advs/' + name
            self.class_file.append((temp_file,advs_dir,label))
        if shuffle:
            random.shuffle(self.class_file)

    def __getitem__(self, idx):
        ori,advs,label =self.class_file[idx]
        if self.transform == None:
            ori = to_tensor(Image.open(ori))
            advs = to_tensor(Image.open(advs))
        else:
            ori = self.transform(Image.open(ori))
            advs = self.transform(Image.open(advs))
        label = torch.tensor(int(label))
        return ori,advs,label

    def __len__(self):
        return len(self.class_file)


class eval_imageNet_datasets(Dataset):
    def __init__(self, filename, shuffle=True):
        print("=====>>>load eval_imageNet_datasets from {}".format(filename))
        ori_dir = filename + '/ori/'
        class_list=os.listdir(ori_dir)
        self.class_file=[]
        for name in class_list:
            temp_file = ori_dir +name
            label = name.split('.')[0].split('_')[-1]
            advs_dir = filename + '/advs/' + name
            self.class_file.append((temp_file,advs_dir,label))
        if shuffle:
            random.shuffle(self.class_file)

    def __getitem__(self, idx):
        ori,advs,label =self.class_file[idx]
        ori = to_tensor(Image.open(ori))
        advs = to_tensor(Image.open(advs))
        label = torch.tensor(int(label))
        return ori,advs,label

    def __len__(self):
        return len(self.class_file)


class generate_ADV_datasets(Dataset):
    def __init__(self, filename,transform,shuffle=True):
        class_list=os.listdir(filename)
        self.class_file=[]
        for name in class_list:
            temp_file=os.path.join(filename,name)
            for item in os.listdir(temp_file):
                img_dir=os.path.join(temp_file,item)
                self.class_file.append((img_dir,name))
        if shuffle:
            random.shuffle(self.class_file)
        self.transform = transform

    def __getitem__(self, idx):
        img,label=self.class_file[idx]
        img = self.transform(Image.open(img))
        label = torch.tensor(int(label))
        return img,label

    def __len__(self):
        return len(self.class_file)

