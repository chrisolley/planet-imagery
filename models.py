import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        self.name = 'ResNet34'
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


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.name = 'VGG19'
        self.vgg = models.vgg19(pretrained=True)
        # freezing parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        # modify the final linear layer
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier = self.vgg.classifier[:6]
        # separate layers into three groups
        layers = list(self.vgg.children())
        self.groups = nn.ModuleList([nn.Sequential(*h) for h in [layers[:2], layers[2:]]])
        self.groups.append(nn.Linear(num_features, num_classes))

    def forward(self, x):
        x = self.groups[0](x)
        x = x.view(x.size(0), -1)
        for group in self.groups[1:]:
            x = group(x)
        return x

    def unfreeze(self,  group_idx: int):
        group = self.groups[group_idx]
        parameters = filter(lambda x: hasattr(x, 'requires_grad'),
                            group.parameters())
        for p in parameters:
            p.requires_grad = True


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.name = 'VGG16'
        self.vgg = models.vgg16(pretrained=True)
        # freezing parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        # modify the final linear layer
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier = self.vgg.classifier[:6]
        # separate layers into three groups
        layers = list(self.vgg.children())
        self.groups = nn.ModuleList([nn.Sequential(*h) for h in [layers[:2], layers[2:]]])
        self.groups.append(nn.Linear(num_features, num_classes))

    def forward(self, x):
        x = self.groups[0](x)
        x = x.view(x.size(0), -1)
        for group in self.groups[1:]:
            x = group(x)
        return x

    def unfreeze(self,  group_idx: int):
        group = self.groups[group_idx]
        parameters = filter(lambda x: hasattr(x, 'requires_grad'),
                            group.parameters())
        for p in parameters:
            p.requires_grad = True