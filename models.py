import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # freezing parameters
        for param in resnet.parameters():
            param.requires_grad = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:8]
        # separate conv layers into two groups
        self.groups = nn.ModuleList([nn.Sequential(*h) for h in [layers[:6], 
                                                                 layers[6:]]])
        # add a linear layer as the classifier
        self.groups.append(nn.Linear(512, num_classes))

    def forward(self, x):
        for group in self.groups[:2]:
            x = group(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        x = self.groups[2](x)
        return x

    def unfreeze(self,  group_idx: int):
        group = self.groups[group_idx]
        parameters = filter(lambda x: hasattr(x, 'requires_grad'),
                            group.parameters())
        for p in parameters:
            p.requires_grad = True
