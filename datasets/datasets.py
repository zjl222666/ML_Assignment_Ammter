import os
from tkinter.tix import Tree
import numpy as np
import json
from PIL import Image
import io
from pathlib import Path
from torchvision.transforms import ColorJitter
import datasets.transforms as T
import torch

class EMDataset():
    def __init__(self, root, train=True, transform=None):
        '''
        the implement here is just set target_type to "full"
        '''
        
        self.base_folder = Path(root)
        path_txt = os.path.join(root, f'{"train" if train else "val"}.txt')

        txt_file = open(path_txt)
        lines = txt_file.readlines()

        metas_names = []
        metas_labels = []
        self.num = 0

        for line in lines:
            secs= line.split('\t')
            name, jsonobj = secs[0], json.loads(secs[1])[0]
            
            w0, h0 = 1000, 1000
            w1, h1 = 0, 0
            for i in jsonobj['points']:
                w0 = min(i[0], w0)
                h0 = min(i[1], h0)
                w1 = max(i[0], w1)
                h1 = max(i[1], h1)
            label = dict()
            label['boxes'] = torch.tensor([w0,h0,w1,h1])

            label['text'] = torch.tensor([10] * 6)
            try:
                assert label['boxes'][0] < label['boxes'][2] and label['boxes'][1] < label['boxes'][3], "position error"
                assert jsonobj['transcription'] != ""
                for id, num in enumerate(jsonobj['transcription']):
                    label['text'][id] = int(num)
                metas_labels.append(label)
                metas_names.append(os.path.join(root, name))
                self.num += 1
            except:
                print(f"failed in load line {line}")
            
        
        self.metas_names = np.string_(metas_names)
        self.metas_labels = metas_labels
        self.initialized = False
        self.transform = transform

    def _init_ceph(self):
        from petrel_client.client import Client as CephClient
        if not self.initialized:
            self.mclient = CephClient()
            self.initialized = True

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        self._init_ceph()
        img_path = str(self.metas_names[idx], encoding='utf-8')
        
        label = self.metas_labels[idx]
        value = self.mclient.Get(img_path)
        img_bytes = np.fromstring(value, np.uint8)
        buff = io.BytesIO(img_bytes)
        with Image.open(buff) as img:
            img = img.convert('RGB')

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label


def build_transfrom(train = True):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if train:
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.ColorJitter(),      
            T.RandomSizeCrop(),
            T.RandomResize([50, 100, 224, 300, 384, 400, 480, 640]),
            T.RandomPad(100),
            T.RandomResize([480], max_size=640),
            normalize,
        ])
    else:
        return T.Compose([
            T.FixPad(),
            T.RandomResize([480], max_size=640),  
            normalize
        ])


def build_dataset(root, train = True, noAug = False):
    transform = build_transfrom(train and not noAug,)
    dataset = EMDataset(root, train, transform=transform)
    return dataset