"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        # def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True)

        # backbone
        self.backbone = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

        # classifier
        # len(classifier) = 5
        self.classifier = list(self.backbone.classifier.children())       
        self.backbone.classifier = nn.Sequential(*self.classifier[:-1])
        self.backbone.classifier.add_module('4', nn.Conv2d(
            in_channels = 256, 
            out_channels = num_classes, 
            kernel_size = 1, 
            stride=1))
                                        
        # aux_classifier
        # len(aux_classifier) = 5
        aux_classifier = list(self.backbone.aux_classifier.children())
        self.backbone.aux_classifier = nn.Sequential(*aux_classifier[:-1])
        self.backbone.aux_classifier.add_module('4', nn.Conv2d(
            in_channels = 256, 
            out_channels = num_classes, 
            kernel_size = 1, 
            stride = 1))

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        # return self.backbone(x)['out'] if aux is not used
        return self.backbone(x)['aux']

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)