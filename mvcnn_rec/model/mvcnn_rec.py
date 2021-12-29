import torch.nn as nn
import torch
import torchvision.models as models
class MVCNNReconstruction(nn.Module):

    def __init__(self):
        """
        """
        super().__init__()
        self.num_features = 12
        self.encoder_image = models.resnet18(pretrained=True)
        self.reconstruction = MVCNNRec()
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,self.num_features)
        )
    def forward(self, x_in):
        """
        x_in in shape [N, 3, H, W]
        B: Batch size
        N: Number of multiple images per shape
        """
        x_score, x_volume = self.reconstruction(x_in[0])
        class_init = self.encoder_image(x_in[0])
        for i in range(x_in.shape[0]-1):
            x_score_temp, x_volume_temp = self.reconstruction(x_in[i+1])
            x_score = torch.cat([x_score, x_score_temp], dim=1)
            x_volume = torch.cat([x_volume, x_volume_temp], dim=1)
            class_init = torch.max(class_init, self.encoder_image(x_in[i+1]))
        m = nn.Softmax(dim=1)
        x_score = m(x_score)
        x_out = torch.sum(torch.mul(x_score, x_volume), dim=1)
        class_out = self.classifier(class_init)
        return x_out, class_out

class MVCNNRec(nn.Module):

    def __init__(self):
        """
        """
        super().__init__()
        self.num_features = 12
        self.part_res = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-4])
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.decoder = MVCNNDecoder()
        self.csn = MVCNNCSN()
    def forward(self, x_in):
        """
        x_in in shape [N, 3, H, W]
        B: Batch size
        N: Number of multiple images per shape
        """
        N = x_in.shape[0]
        out = self.part_res(x_in)
        out = self.encoder(out).view(N,392,2,2,2)
        c_1, c_2  = self.decoder(out)
        c = torch.cat([c_1, c_2], dim=1)
        return self.csn(c), c_2

class MVCNNDecoder(nn.Module):
    def __init__(self):
        """
        """
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=392, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()

        )
        self.model_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
    def forward(self, x_in):
        """
        x_in in shape [N, 3, H, W]
        B: Batch size
        N: Number of multiple images per shape
        """
        out = self.model_1(x_in)
        return out, self.model_2(out)
class MVCNNCSN(nn.Module):

    def __init__(self):
        """
        """
        super().__init__()

        self.conv1 =nn.Sequential(
            nn.Conv3d(in_channels=9, out_channels=9, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=9, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, x_in):
        """
        x_in in shape [N, 3, H, W]
        B: Batch size
        N: Number of multiple images per shape
        """
        o_1 = self.conv1(x_in)
        o_5 = o_1
        o_2 = self.conv2(o_1)
        o_5 = torch.cat([o_5, o_2], dim=1)
        o_3 = self.conv3(o_2)
        o_5 = torch.cat([o_5, o_3], dim=1)
        o_4 = self.conv4(o_3)
        o_5 = torch.cat([o_5, o_4], dim=1)
        o_5 = self.conv5(o_5)
        return o_5
