from turtle import forward, shape
import  torch
import torch.nn as nn 
from timm.models.factory import create_model
import torch.nn.functional as F
from torchvision.transforms.functional import crop as imgCrop
import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list_fix_size
from models.detr import build_detr
import util.box_ops as box_ops
from util.box_ops import center_to_crop_box

class TSTT(nn.Module):
    def __init__(self, box_detector, text_classifier) -> None:
        super().__init__()
        self.box_detector = box_detector
        self.text_classifier = text_classifier

    def forward(self, x):
        B, C, W, H = x.tensors.shape
        proposed_boxes = self.box_detector(x)
        cropped_x = []
        for batch, box in zip(x.tensors, proposed_boxes):
            box = (center_to_crop_box(box.squeeze()) * torch.tensor([H, W, H, W]))
            cropped_x.append(imgCrop(batch, *box[..., (1,0,3,2)].int()))
            
            # show = T.ToPILImage(cropped_x[-1])
        
        cropped_x = nested_tensor_from_tensor_list_fix_size(cropped_x)

        output = self.text_classifier(cropped_x.detach())

        return proposed_boxes, output




class TSTTLoss(nn.Module):
    def __init__(self, num_classes = 10, eos_coef = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def forward(self, output, target, box_coef = 1.0, text_coef = 1.0, giou_coef = 1.0, metric_logger = None):
        boxes, text = output
        boxes_target = target['boxes']
        text_target = target['text']

        assert len(boxes.shape) != 2 or boxes.shape != boxes_target.shape, "shape boxes output is not allign"
        assert len(text.shape) != 3 or text.shape != text_target.shape, "shape text classifier is not allign"
        
        loss_bbox = F.l1_loss(boxes.sigmoid(), boxes_target, reduction='sum')

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(boxes),
        #     box_ops.box_cxcywh_to_xyxy(boxes_target)))
        # loss_giou = loss_giou.sum()


        loss_text = F.cross_entropy(text.view(-1,self.num_classes + 1), text_target.view(-1) ,self.empty_weight)
        

        if not metric_logger is None:
            metric_logger.update(loss_bbox = loss_bbox)
            metric_logger.update(loss_text = loss_text)
        return loss_bbox * box_coef + loss_text * text_coef # + loss_giou * giou_coef


def build(args):
    box_detector = build_detr(1, 4, args=args)
    text_classifier = build_detr(6, 11, args=args)

    model = TSTT(box_detector, text_classifier)

    loss = TSTTLoss()

    return model, loss

