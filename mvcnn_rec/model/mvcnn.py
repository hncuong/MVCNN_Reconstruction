import torch.nn as nn
import torch
import torchvision.models as models

class MVCNN(nn.Module):
    """
    MVCNN for Classification
    """
    def __init__(self, num_classes=13):
        """ Arguments:
        - num_classes: number of output classes (default 13)
        """
        super().__init__()
        self.num_classes = num_classes
        # TODO: Take chilren [:-1] here
        self.encoder_image = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512,self.num_classes)
        )
    
    def forward(self, x_in):
        """
        Doing multiviews forward part (for single views just have first dim=1)
        """
        class_init = self.encoder_image(x_in[0]) # 512, 1, 1 for (224 x 224) images
        for i in range(x_in.shape[0]-1):
            class_init = torch.max(class_init, self.encoder_image(x_in[i+1]))
        class_init = class_init.view(class_init.shape[0], -1)
        class_out = self.classifier(class_init)
        return class_out
