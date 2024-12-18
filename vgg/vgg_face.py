import torch
from torch import nn
from collections import OrderedDict
from warnings import filterwarnings

filterwarnings("ignore")

KERNEL_SIZE = (3,3)

class VGGFACE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1_1 = nn.Conv2d(3, 64, KERNEL_SIZE, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, KERNEL_SIZE, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, KERNEL_SIZE, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, KERNEL_SIZE, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, KERNEL_SIZE, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, KERNEL_SIZE, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, KERNEL_SIZE, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, KERNEL_SIZE, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1)

        self.fc6 = nn.Linear(49*512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.features = [
            self.conv1_1, self.relu,
            self.conv1_2, self.relu,
            self.maxpool,
            self.conv2_1, self.relu,
            self.conv2_2, self.relu,
            self.maxpool,
            self.conv3_1, self.relu,
            self.conv3_2, self.relu,
            self.conv3_3, self.relu,
            self.maxpool,
            self.conv4_1, self.relu,
            self.conv4_2, self.relu,
            self.conv4_3, self.relu,
            self.maxpool,
            self.conv5_1, self.relu,
            self.conv5_2, self.relu,
            self.conv5_3, self.relu,
            self.maxpool
        ]

        self.classifier = [
            self.fc6, self.relu,
            self.fc7, self.relu,
            self.fc8
        ]

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        return self.classifier(x)
    
    def embeddings(self, x:torch.Tensor):
        return self.features(x).flatten().detach().numpy()
    
MODEL_FACE = VGGFACE()
MODEL_FACE.load_state_dict(torch.load("models/vgg_face_dag.pth"), strict=True)

if __name__ == "__main__":
    print(MODEL_FACE.state_dict().keys())