import os
from tkinter.tix import Tree
import numpy as np
import json
from PIL import Image
import io
from pathlib import Path
import torchvision.transforms as T
import torch
import random
import torchvision.transforms.functional as F
class SelfData():
    def __init__(self, root, transform=None, length = 1000):
        '''
        the implement here is just set target_type to "full"
        '''
        root = Path(root)
        root = root / 'selfdata/numbers'
        img_path = []
        img_label = []
        self.num = length
        for path,dir_list,file_list in  os.walk(root): 
            for file in file_list:
                img_path.append(root / file)
                img_label.append(int(file[0]))
                    
        
        self.metas_names = np.string_(img_path)
        self.metas_labels = np.int_(img_label)

        self.initialized = False
        self.transform = transform
        self.resize = RandomResize()
        # print(f"===={self.metas_names}====")
        # print(f"====={self.metas_labels}======")

    def _init_ceph(self):
        from petrel_client.client import Client as CephClient
        if not self.initialized:
            self.mclient = CephClient()
            self.initialized = True

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        self._init_ceph()
        label = {}
        label['text'] = torch.tensor([10] * 6)
        imgs = []
        lenght = random.randint(3,6)
        randid = []
        for i in range(lenght):
            randid.append(random.randint(0, len(self.metas_names) - 1))
            label['text'][i] = self.metas_labels[randid[-1]]
        
        for id in randid:       
            img_path = str(self.metas_names[id], encoding='utf-8')
            value = self.mclient.Get(img_path)
            img_bytes = np.fromstring(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as img:
                img = img.convert('RGB')
            img = self.resize(img, idx)
            img = self.transform(img)
            imgs.append(img)

        pad_x_1 = random.randint(0, 200)
        pad_y_1 = random.randint(0, 200)
        pad_x = 200 - pad_x_1
        pad_y = 200 - pad_y_1
        img = torch.cat(imgs, -1)
        img = F.pad(img, (pad_x, pad_y,pad_x_1,pad_y_1))

        return img, label


class RandomResize(object):
    def __init__(self, size = [75, 100, 130, 150]):
        self.size = size

    def __call__(self, img, idx):
        size = self.size[idx % 4]
        return F.resize(img, [size,size])

def build_transfrom():
    return T.Compose([
        T.ColorJitter( brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5,
            hue = 0.5),
        T.RandomCrop(100, pad_if_needed = True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_self_dataset(root, length):
    transform = build_transfrom()
    dataset = SelfData(root, transform=transform, length=length)
    return dataset