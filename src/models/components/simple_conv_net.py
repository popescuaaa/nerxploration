from torch import nn
from typing import Tuple


class SimpleConvNet(nn.Module):
    """
    This a simple conv net for MNIST + FashionMNIST.
    The architecture is taken from this link:
    - https://machinelearningknowledge.ai/pytorch-conv2d-explained-with-examples/#:~:text=It%20is%20a%20simple%20mathematical,one%20output%20of%20that%20operation.

    Each conv layer is defined by:
        - in channels size
        - out channels size
        - kernel size
        - stride
        - padding
    """

    def __init__(
            self,
            layer1: int = 32,
            layer2: int = 64,
            layer3: int = 128,
            layer4: int = 256,
            layer5: int = 10
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=layer1,
                out_channels=layer2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=layer2,
                out_channels=layer3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.3)
        )

        self.conv = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        # kernel * kernel * layer3
        self.linear1 = nn.Linear(3 * 3 * layer3, layer4, bias=True)
        self.linear2 = nn.Linear(layer4, layer5, bias=True)

        self.fc1 = nn.Sequential(
            self.linear1,
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            self.linear2,
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    _ = SimpleConvNet()
