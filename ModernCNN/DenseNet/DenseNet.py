import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


# blk = DenseBlock(2, 3, 10)
# X = torch.randn(4, 3, 8, 8)
# Y = blk(X)
# print(Y.shape)

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)).to(device)

transform = transforms.Compose([
    Resize([96, 96]),
    ToTensor()
])
mnist_training = datasets.MNIST(
    root="../data",
    train=True,
    transform=transform,
    download=False
)

mnist_test = datasets.MNIST(
    root="../data",
    train=False,
    transform=transform,
    download=False
)


BATCH_SIZE = 256
lr = 0.1
epochs = 20

train_dataloader = DataLoader(mnist_training, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

# net = xxxNet().to(device)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    print(
        f"epoch {epoch} \n---------------------"
    )

    for batch, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss:{loss:>7f} [{current:>5d}/ 60000]")

    with torch.no_grad():
        acc = 0
        total = 0
        for (image, label) in test_dataloader:
            image, label = image.to(device), label.to(device)
            output = net(image)
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            acc += (pred == label).sum()

        print(f"test: acc {100 * acc / total}")


