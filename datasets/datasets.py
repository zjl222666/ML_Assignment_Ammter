import os
from tkinter.tix import Tree
import numpy as np
import json
from PIL import Image
import io
from pathlib import Path
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
            metas_names.append(os.path.join(root, name))
            w0, h0 = jsonobj['points'][0][0], jsonobj['points'][0][1]
            w1, h1 = jsonobj['points'][2][0], jsonobj['points'][2][1]
            label = dict()
            label['boxes'] = torch.tensor([w0,h0,w1,h1])

            label['text'] = torch.tensor([11] * 6)
            try:
                for id, num in enumerate(jsonobj['transcription']):
                    label['text'][id] = int(num)
                metas_labels.append(label)
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
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if train:
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize,
        ])
    else:
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize
        ])


def build_dataset(root, train = True):
    transform = build_transfrom(train)
    dataset = EMDataset(root, train, transform=transform)
    return dataset