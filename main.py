# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.mobilenet_v2.networks import MobileNet_v2
from datasets import get_loaders


def train(epoch, model, optimizer, criterion, train_loader, device):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    print("Epoch: {}, Loss: {:.3f}, Acc: {:.3f}".format(
        epoch,
        train_loss / total,
        100. * correct / total))

def test():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model", type=str, default="mobilenet_v2")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--dataroot", type=str, default="/tmp/data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, test_loader, n_classes = get_loaders(dataset=args.dataset,
                                                       root=args.dataroot,
                                                       batch_size=args.batch_size)

    if args.model == "mobilenet_v2":
        model = MobileNet_v2(n_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        for epoch in range(args.n_epochs):
            train(epoch, model, optimizer, criterion, train_loader, device)

if __name__ == "__main__":
    main()
