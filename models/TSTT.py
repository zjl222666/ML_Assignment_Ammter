from turtle import forward, shape
import  torch
import torch.nn as nn 
from timm.models.factory import create_model
import torch.nn.functional as F
from torchvision.transforms.functional import crop as imgCrop
from torchvision.transforms.functional import resize
import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list
from models.detr import build_detr
import util.box_ops as box_ops
from util.box_ops import center_to_crop_box
from util.misc import NestedTensor
from torch.utils.checkpoint import checkpoint



class TSTT(nn.Module):
    def __init__(self, box_detector, text_classifier) -> None:
        super().__init__()
        self.box_detector = box_detector
        self.text_classifier = text_classifier

    def forward(self, x, need_cropped = False):
        if isinstance(x, torch.Tensor):
            H, W = x.shape[-2:]
        else:
            H, W = x.tensors.shape[-2:]

        # if self.train:
        #     proposed_boxes = checkpoint(self.box_detector, x).sigmoid()
        # else:
        #     
        proposed_boxes = self.box_detector(x).sigmoid()
        
        
        if isinstance(x, NestedTensor):
            x = x.tensors

        # print(f"=========={x.shape, proposed_boxes.shape}=========")
        cropped_x = []
        for batch, box in zip(x, proposed_boxes[-1]):
            box = (center_to_crop_box(box.squeeze())) * torch.tensor([W, H, W, H]).to(x.device)
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(W - box[0], box[2])
            box[3] = min(H - box[1], box[3])
            box[2] = max(1, box[2])
            box[3] = max(1, box[3])
            cropped_x.append(resize(imgCrop(batch, *box[(1,0,3,2),].int()).detach(), 200))
            
            # show = T.ToPILImage(cropped_x[-1])

        cropped_x = nested_tensor_from_tensor_list(cropped_x)
        # print(cropped_x)
        # if self.train:
        #     output = checkpoint(self.text_classifier, cropped_x)
        # else:
        output = self.text_classifier(cropped_x)
        if need_cropped:
            return proposed_boxes[-1].squeeze(1), output[-1], cropped_x
        return proposed_boxes[-1].squeeze(1), output[-1]

class class_Loss(nn.Module):
    def __init__(self, num_classes = 10, eos_coef = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, output, target, box_coef = 1.0, text_coef = 1.0, giou_coef = 1.0, box =True, metric_logger = None):

        text = output
        text_target = torch.stack([t['text'] for t in target])
        loss_text = F.cross_entropy(text.view(-1,self.num_classes + 1), text_target.view(-1) ,self.empty_weight)
        return loss_text

class TSTTLoss(nn.Module):
    def __init__(self, num_classes = 10, eos_coef = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def forward(self, output, target, box_coef = 1.0, text_coef = 1.0, giou_coef = 1.0, box = True,metric_logger = None):
        boxes, text = output
        boxes_target = torch.stack([t['boxes'] for t in target])
        text_target = torch.stack([t['text'] for t in target])
        if box:
            text_coef = 0
        else:
            box_coef = giou_coef = 0

        assert len(boxes.shape) == 2 and boxes.shape == boxes_target.shape, "shape boxes output is not allign"
        assert len(text.shape) == 3 and text.shape[:-1] == text_target.shape, f"{text.shape, text_target.shape} shape text classifier is not allign"
        assert text.shape[-1] == 11

        loss_bbox = F.l1_loss(boxes, boxes_target, reduction='sum')

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(boxes),
            box_ops.box_cxcywh_to_xyxy(boxes_target)))
        loss_giou = loss_giou.sum()
        loss_text = F.cross_entropy(text.view(-1,self.num_classes + 1), text_target.view(-1) ,self.empty_weight)


        if not metric_logger is None:
            metric_logger.update(loss_bbox = loss_bbox.item())
            metric_logger.update(loss_text = loss_text.item())
            metric_logger.update(loss_giou = loss_giou.item())
        return loss_bbox * box_coef + loss_text * text_coef + loss_giou * giou_coef


def build(args1, args2):
    box_detector = build_detr(1, 4, False, args=args1)
    text_classifier = build_detr(6, 11, args2.multi_final, args=args2)
    if not args1.one_stage:
        model = TSTT(box_detector, text_classifier)
        loss = TSTTLoss()
    else:
        model = text_classifier
        loss = class_Loss()

    return model, loss

