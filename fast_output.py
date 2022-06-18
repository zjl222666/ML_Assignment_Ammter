import torch
from datasets.datasets import build_transfrom
import os
from pathlib import Path
import numpy as np
from petrel_client.client import Client as CephClient
from PIL import Image
from torchvision.transforms import ToPILImage, ColorJitter
import io

test_path = "/mnt/cache/zhengjinliang/ElectricityMeter/real_test"
test_path = Path(test_path)
img_path = []
for path,dir_list,file_list in  os.walk(test_path): 
    for file in file_list:
        img_path.append(test_path / file)

img_path = np.string_(img_path)
mclient = CephClient()
transform = build_transfrom(False)
def get_img(img_path):
    img_path = str(img_path, encoding='utf-8')
    value = mclient.Get(img_path)
    img_bytes = np.fromstring(value, np.uint8)
    buff = io.BytesIO(img_bytes)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    img, _ = transform(img, {"boxes": torch.tensor([0,0,0,0])})
    
    c, h, w = img.shape
    return img

@torch.no_grad()
def run(img_path, vis = False):
    img = get_img(img_path)
    img = img.unsqueeze(0)
    output = model(img, True)
    box, final, cropped = output
    text_class = text_classifier(cropped.tensors)
    if vis:
        print(final.softmax(-1).max(-1).indices)
        print(text_class.softmax(-1).max(-1).indices)
        print(box)
        display(ToPILImage()(img.squeeze()))
        display(ToPILImage()(cropped.tensors[0]))

    final = final.softmax(-1).max(-1).indices.squeeze()

    number = 0
    for i in final:
        number = number * 10 + (i % 10)
    
    if final[-1] == 10: number *= 10
    return number.item() / 10
def write(id, num):
    with open("result.txt", "a+") as f:
        f.write(f"{id},{num}\n")
from models import build_model
from easydict import EasyDict
args = dict(backbone = "resnet18", position_embedding = 'sine', masks = False, dilation = False,
    enc_layers=3, dec_layers=3, dim_feedforward=512, hidden_dim=192, dropout=0.1, nheads=6,
    lr_backbone = 1e-4,    pre_norm=False, one_stage = False, multi_final = True
 )
args = EasyDict(args)
model = build_model(args)
model = model[0]
ckpt = torch.load("/mnt/cache/zhengjinliang/Ammeter/exp/good_2/ckpt2/checkpoint.pth", map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()
args = dict(backbone = "resnet18", position_embedding = 'sine', masks = False, dilation = False,
    enc_layers=3, dec_layers=3, dim_feedforward=512, hidden_dim=192, dropout=0.1, nheads=6,
    lr_backbone = 1e-4,    pre_norm=False, one_stage = True, multi_final = True
 )
args = EasyDict(args)
text_classifier = build_model(args)[0]
ckpt = torch.load("/mnt/cache/zhengjinliang/Ammeter/exp/pretrain_2/ckpt2/checkpoint.pth", map_location='cpu')
text_classifier.load_state_dict(ckpt['model'])
for id, path in enumerate(img_path):

    name = str(path).split('/')[-1].split('.')[0]
    ans = run(path)
    write(name, ans)
    print(f"{id} is  ok")