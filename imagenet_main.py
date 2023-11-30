from train_funcs import FFTrainer
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


def find_imagenet_mean_std(batch_size: int = 1000):
    print('transforms')
    transform = transforms.Compose([transforms.ToTensor()])
    print('imagenet dataset')
    train_data = ImageNet(r'./imagenet_data', transform=transform)
    print('dataloader')
    train_loader = DataLoader(train_data, batch_size=batch_size)
    # labels = torch.tensor([])

    mean = 0
    std = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('starting iteration')
    for i, (batch, label) in enumerate(train_loader):
        print(f'processing batch {i}')
        batch = batch.to(device)
        # labels = torch.unique(torch.cat((labels, label), dim=0))
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    return (mean / len(train_loader.dataset)).item(), (std / len(train_loader.dataset)).item()


def create_imagenet_dataset(batch_size: int = 25):
    # mean, std, _ = find_imagenet_mean_std()
    mean, std = [0.485, 0.456, 0.406],  [0.229, 0.224, 0.225]
    print(f'Mean & STD: ({mean},{std})')
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(200),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean[0], mean[1], mean[2],), (std[0], std[1], std[2],)),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    train_data = ImageNet(root=r'./imagenet_data', split='train', transform=transform)
    test_data = ImageNet(root=r'./imagenet_data', split='val', transform=transform)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    ff_mod = FFTrainer(plot_title='ImageNet',
                       save_file_dir='./imagenet_out',
                       save_file_name='imagenet',
                       batch_size_params=(100, 200, 100),
                       ff_thresh_params=(0.5, 4.0, 0.5),
                       epoch_params=(60, 80, 20),
                       lr_params=(0.03, 0.04, 0.02),
                       layers=(120000, 1000, 1000, 1000))

    ff_mod.train(create_imagenet_dataset, 1000)
