from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from resnet_mnist import train_resnet

import torch


def find_cifar100_mean_std(batch_size: int = 1000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = CIFAR100(root='./cifar10_data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    mean = 0
    std = 0

    for batch, label in train_loader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    return (mean / len(train_loader.dataset)).tolist(), (std / len(train_loader.dataset)).tolist()


def create_cifar100_dataset(train_batch_size: int = 25, test_batch_size: int = 25):
    mean, std = find_cifar100_mean_std()
    print(f'Mean & STD: ({mean},{std})')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((*mean,), (*std,))])

    train_data = CIFAR100(root='./cifar10_data', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='./cifar10_data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=train_batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


if __name__ == '__main__':
    train_data, test_data = create_cifar100_dataset(768, 768)
    train_resnet(save_file_dir=r'./resnet_cifar100',
                 save_file_name=r'resnet_cifar100',
                 plot_title='CIFAR 100',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 optimizer_func=torch.optim.Adam,
                 optimizer_kwargs={'lr': 0.001},
                 train_dataloader=train_data,
                 test_dataloader=test_data,
                 total_epochs=200,
                 total_classes=100)
