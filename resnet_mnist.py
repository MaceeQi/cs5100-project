
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from tqdm import tqdm
from usage import TrackUsage
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from traceback import format_exc

import torch, os, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc


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
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                    ])

    train_data = MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    test_data = MNIST(root='./mnist_data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=train_batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


def checkpoint(save_file_dir: str,
               save_file_name: str,
               plot_title: str,
               best_y_true: list = [],
               best_y_pred: list = []):
    overall_auc = None

    try:
        if best_y_true and best_y_pred:
            uniq_labels = range(len(set(best_y_true)))
            bin_y_true = label_binarize(best_y_true, classes=list(range(len(uniq_labels))))
            bin_y_pred = label_binarize(best_y_pred, classes=list(range(len(uniq_labels))))

            for i in range(len(uniq_labels)):
                fpr, tpr, _ = roc_curve(bin_y_true[:, i], bin_y_pred[:, i])
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Class {i} ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{plot_title} ROC Curve - Class {i}')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(save_file_dir,
                                         '{0}_roc_curve_class_{1}.png'.format(save_file_name, i)))
                plt.close()
                del fpr, tpr, roc_auc

            overall_auc = roc_auc_score(bin_y_true, bin_y_pred, multi_class="ovr")

            conf_matrix = confusion_matrix(best_y_true, best_y_pred)
            cm_df = pd.DataFrame(conf_matrix, index=[i for i in uniq_labels],
                                 columns=[i for i in uniq_labels])

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt='d')
            plt.title(f'{plot_title} Confusion Matrix')
            plt.ylabel('Actual Labels')
            plt.xlabel('Predicted Labels')
            plt.savefig(os.path.join(save_file_dir,
                                     '{0}_confusion_matrix.png'.format(save_file_name)))
            plt.close()

            del cm_df, conf_matrix, bin_y_pred, bin_y_true, uniq_labels
    except Exception as e:
        print('Failed Recording Checkpoint -', e)
        format_exc()
    finally:
        plt.close('all')
        gc.collect()

    return overall_auc


def save_csv(save_file_dir: str, save_file_name: str, total_trials: list):
    try:
        trials_df = pd.DataFrame(total_trials)
        trials_df.to_csv(os.path.join(save_file_dir,
                                      '{0}_trials.csv'.format(save_file_name)), index=False)
    except Exception as e:
        print('Failed Recording Checkpoint -', e)
        format_exc()


def train_resnet(save_file_dir: str,
                 save_file_name: str,
                 plot_title: str,
                 loss_func,
                 optimizer_func,
                 optimizer_kwargs,
                 train_dataloader,
                 test_dataloader,
                 total_epochs: int,
                 total_classes: int):
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_mod = resnet152(pretrained=True)
    resnet_mod.fc = torch.nn.Linear(resnet_mod.fc.in_features, total_classes)
    optimizer = optimizer_func(resnet_mod.parameters(), **optimizer_kwargs)
    resnet_mod = resnet_mod.to(device)
    best_epoch = float('-inf')
    e_stats = []

    for epoch in tqdm(range(total_epochs), desc="Epochs"):
        tracker = TrackUsage()
        tracker.start()
        resnet_mod.train()
        last_loss = 0
        y_pred, y_true = [], []
        e_stat = {'epoch': epoch,
                  'loss': None,
                  'accuracy': None,
                  'f1_score': None,
                  'precision': None,
                  'recall': None,
                  'avg_cpu_perc': None,
                  'avg_cpu_memory': None,
                  'max_cpu_memory': None,
                  'cpu_memory_start': None,
                  'avg_gpu_perc': None,
                  'avg_gpu_memory': None,
                  'max_gpu_memory': None,
                  'gpu_memory_start': None,
                  'time_duration': None,
                  'overall_auc': None
                  }

        for data, labels in train_dataloader:
            labels = labels.to(device)
            outputs = resnet_mod(data.to(device))
            loss = loss_func(outputs, labels)
            last_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        resnet_mod.eval()

        with torch.no_grad():
            for data, labels in test_dataloader:
                y_pred.extend(resnet_mod(data.to(device)).argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())

        e_stat['loss'] = last_loss
        e_stat['accuracy'] = accuracy_score(y_true, y_pred)
        e_stat['f1_score'] = f1_score(y_true, y_pred, average='macro')
        e_stat['precision'] = precision_score(y_true, y_pred, average='macro')
        e_stat['recall'] = recall_score(y_true, y_pred, average='macro')

        tracker.stop()
        e_stat['avg_cpu_perc'] = tracker.avg_cpu_perc
        e_stat['avg_cpu_memory'] = tracker.avg_cpu_memory
        e_stat['max_cpu_memory'] = tracker.max_cpu_memory
        e_stat['cpu_memory_start'] = tracker.cpu_memory_start
        e_stat['avg_gpu_perc'] = tracker.avg_gpu_perc
        e_stat['avg_gpu_memory'] = tracker.avg_gpu_memory
        e_stat['max_gpu_memory'] = tracker.max_gpu_memory
        e_stat['gpu_memory_start'] = tracker.gpu_memory_start
        e_stat['time_duration'] = tracker.time_duration
        acc = e_stat['accuracy']
        e_stats.append(e_stat)

        tqdm.write(
            f'Epoch {epoch + 1}/{total_epochs} - Loss: {last_loss:.4f}, Accuracy: {acc:.4f}')

        if best_epoch < e_stat['accuracy']:
            best_epoch = e_stat['accuracy']
            e_stat['overall_auc'] = checkpoint(save_file_dir, save_file_name, plot_title, y_true, y_pred)
            save_csv(save_file_dir, save_file_name, e_stats)
            print('\nBest Model:', e_stat)
            torch.save(resnet_mod.state_dict(),
                       os.path.join(save_file_dir,
                                    '{0}_model.pth'.format(save_file_name)))
        else:
            e_stat['overall_auc'] = checkpoint(save_file_dir, save_file_name, plot_title)
            save_csv(save_file_dir, save_file_name, e_stats)


if __name__ == '__main__':
    train_data, test_data = create_mnist_dataset()
    train_resnet(save_file_dir=r'./resnet_mnist',
                 save_file_name=r'resnet_mnist',
                 plot_title='MNIST',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 optimizer_func=torch.optim.SGD,
                 optimizer_kwargs={'lr': 0.001, 'momentum': 0.9},
                 train_dataloader=train_data,
                 test_dataloader=test_data,
                 total_epochs=10,
                 total_classes=10)
