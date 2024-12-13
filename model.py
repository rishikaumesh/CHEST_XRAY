import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=2, dropout_rate=0.3):
        super(MyModel, self).__init__()

        # Load the pretrained model and set the desired backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Modify conv1 for single-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Modify conv1 for single-channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            # Modify first convolutional layer for single-channel input
            self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            # EfficientNet doesn't use a typical conv1 layer; it adapts to single-channel automatically
        elif backbone == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            # Modify the first convolutional layer for single-channel input
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            raise ValueError("Invalid backbone specified. Choose from 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', or 'mobilenet_v2'.")

        # Unfreeze all layers for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # Replace the final classifier layer for each model architecture
        if backbone in ['resnet18', 'resnet50']:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.model.fc.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif backbone == 'densenet121':
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.model.classifier.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif backbone == 'efficientnet_b0':
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.model.classifier[1].in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif backbone == 'mobilenet_v2':
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.model.classifier[1].in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)
