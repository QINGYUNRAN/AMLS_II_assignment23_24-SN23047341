import torch
import torch.nn as nn
import timm

# print(timm.list_models("vit*"))


class ViTBase16(nn.Module):
    """
            Initialize the Vision Transformer (ViT) Base 16 class.

            Parameters:
            - n_classes (int): The number of classes for the output layer.
            - pretrained (bool): Whether to use a pre-trained model. Default is True.
            """

    def __init__(self, n_classes, pretrained=True):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
