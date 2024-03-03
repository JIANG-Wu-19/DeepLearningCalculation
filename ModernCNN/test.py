'''
这是一个面向数据集Fashion-MNIST/MNIST的测试框架，只要将封装好的模型import并实例化即可
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize

# from xxxNet import xxxNet

transform = transforms.Compose([
    Resize([224, 224]),
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

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
lr = 0.01
epochs = 20

train_dataloader = DataLoader(mnist_training, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

# net = xxxNet().to(device)
net = nn.Sequential()


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
