import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from MLP import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_training = datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=ToTensor(),
    download=False
)

mnist_test = datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=ToTensor(),
    download=False
)

BATCH_SIZE = 256
lr = 0.01
epochs = 20

loss_fn = nn.CrossEntropyLoss()

train_dataloader = DataLoader(mnist_training, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)


# class MLP(nn.Module):
#     def __init__(self, input_num):
#         super(MLP, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_num, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10),
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


def train(train_iter, model, loss_fn, optimizer):
    print(f"train on {device}")

    model = model.to(device)
    size = len(train_iter.dataset)

    for batch, (X, y) in enumerate(train_iter):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


def test(test_iter, model, loss_fn):
    model = model.to(device)
    size = len(test_iter.dataset)
    num_batches = len(test_iter)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches

    correct /= size

    print(f"test error:\n accuracy:{(100 * correct):>0.1f}%,avg loss: {test_loss:>8f} \n")


model = MLP(28 * 28)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for t in range(epochs):
    print(f"epoch{t+1} \n ---------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader,model,loss_fn)

print("done")
