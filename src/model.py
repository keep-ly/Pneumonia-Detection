import torch
import torch.nn as nn
from torchvision import models


def create_model(config: dict):
    """根据配置创建模型（支持 ResNet50 / EfficientNet-B0 / 自定义 CNN）。"""
    architecture = config["model"]["architecture"]
    pretrained = config["model"]["pretrained"]
    dropout_rate = config["model"]["dropout_rate"]

    if architecture == "resnet50":
        model = _create_resnet50(pretrained, dropout_rate)
    elif architecture == "efficientnet-b0":
        model = _create_efficientnet_b0(pretrained, dropout_rate)
    elif architecture == "custom_cnn":
        model = _create_custom_cnn(dropout_rate)
    else:
        raise ValueError(f"不支持的模型架构: {architecture}，可选: resnet50 / efficientnet-b0 / custom_cnn")

    return model


def _create_resnet50(pretrained: bool, dropout_rate: float):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 1),
    )
    return model


def _create_efficientnet_b0(pretrained: bool, dropout_rate: float):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 1),
    )
    return model


def _create_custom_cnn(dropout_rate: float):
    """轻量自定义 CNN，适合快速实验。"""
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, 1),
    )
    return model
