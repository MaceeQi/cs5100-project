from train_funcs import FFTrainer
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
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
                                    transforms.Normalize((*mean,), (*std,)),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    train_data = CIFAR100(root='./cifar10_data', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='./cifar10_data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=train_batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


if __name__ == '__main__':
    ff_mod = FFTrainer(plot_title='CIFAR 10',
                       save_file_dir='./cifar10_out',
                       save_file_name='cifar10',
                       batch_size_params=(250, 500, 500),
                       ff_thresh_params=(4.0, 10.0, 0.5),
                       epoch_params=(60, 80, 20),
                       lr_params=(0.03, 0.04, 0.02),
                       layers=(3072, 3072, 3072))

    ff_mod.train(create_cifar100_dataset, 100)
