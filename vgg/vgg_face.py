import torch
from torch import nn
from warnings import filterwarnings
from torchvision.transforms import ToTensor, Resize, Normalize, Compose

filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self._features = [
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
            self.maxpool,
            nn.Flatten(start_dim=0)
        ]

        self._classifier = [
            self.fc6, self.relu,
            self.fc7, self.relu,
            self.fc8
        ]

        self._embedder = [
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
            self.maxpool,
            nn.Flatten(start_dim=0),
            self.fc6,
        ]
        self.transform = Compose([ToTensor() ,Resize((224, 224)), Normalize(mean=(93.59396362304688/255, 104.76238250732422/255, 129.186279296875/255), std=(1, 1, 1))])
    def features(self, x):
        x = self.transform(x)
        x = x.to(DEVICE)
        for layer in self._features:
            x = layer(x)
        return x
    def classifier(self, x):
        for layer in self._classifier:
            x = layer(x)
        return x
    def embedder(self, x):
        x = self.transform(x)
        x = x.to(DEVICE)
        for layer in self._embedder:
            x = layer(x)
        return x
    def forward(self, x:torch.Tensor):
        x = self.features(x)
        return self.classifier(x)
    def embeddings(self, x:torch.Tensor):
        return self.embedder(x).cpu().flatten().detach().numpy()
    __call__ = embeddings
    
MODEL_FACE = VGGFACE()
MODEL_FACE.load_state_dict(torch.load("models/vgg_face_dag.pth"), strict=True)
MODEL_FACE.to(DEVICE)

if __name__ == "__main__":
    print(MODEL_FACE.state_dict().keys())