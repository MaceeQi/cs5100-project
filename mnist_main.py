from train_funcs import FFTrainer
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


def find_mnist_mean_std(batch_size: int = 1000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    mean = 0
    std = 0

    for batch, _ in train_loader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    return (mean / len(train_loader.dataset)).item(), (std / len(train_loader.dataset)).item()


def create_mnist_dataset(train_batch_size: int = 25, test_batch_size: int = 25):
    mean, std = find_mnist_mean_std()
    print(f'Mean & STD: ({mean},{std})')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,)),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    train_data = MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    test_data = MNIST(root='./mnist_data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=train_batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


if __name__ == '__main__':
    ff_mod = FFTrainer(plot_title='MNIST',
                       save_file_dir='./mnist_out',
                       save_file_name='mnist',
                       batch_size_params=(1000, 1500, 500),
                       ff_thresh_params=(4.0, 10.0, 0.5),
                       epoch_params=(60, 80, 20),
                       lr_params=(0.03, 0.04, 0.02),
                       layers=(784, 500, 500, 500))

    ff_mod.train(create_mnist_dataset, 10)
