from usage import TrackUsage
from traceback import format_exc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from ff_model import FFLinear, FFAdamParams, FFSupervisedVision
from tqdm import tqdm

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import gc
import torch
import numpy as np
import itertools


class FFTrainer(object):
    def __init__(self,
                 plot_title: str,
                 save_file_dir: str,
                 save_file_name: str,
                 batch_size_params: (list, tuple),
                 ff_thresh_params: (list, tuple),
                 epoch_params: (list, tuple),
                 lr_params: (list, tuple),
                 layers: (list, tuple)):

        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)

        self.plot_title = plot_title
        self.save_file_dir = save_file_dir
        self.save_file_name = save_file_name
        self.batch_size_params = batch_size_params
        self.ff_thresh_params = ff_thresh_params
        self.lr_params = lr_params
        self.epoch_params = epoch_params
        self.layers = layers
        self.total_classes = None
        self.device = None

    def train(self, dataset_func, total_classes: int):
        ff_gen = self.__gen_ff_params()

        total_iterations = len(range(*self.batch_size_params)) * len(np.arange(*self.lr_params)) * len(
            range(*self.epoch_params)) * len(ff_gen)
        trials = []
        best_trial = float('-inf')
        torch.manual_seed(3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_classes = total_classes

        with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
            for batch_size in range(*self.batch_size_params):
                train, test = dataset_func(batch_size)

                for lr_param in np.arange(*self.lr_params):
                    adam_params = FFAdamParams(lr=lr_param)

                    for epochs in range(*self.epoch_params):
                        for param_combination in ff_gen:
                            trials, best_trial = self.__run_trial(train,
                                                                  test,
                                                                  adam_params,
                                                                  batch_size,
                                                                  lr_param,
                                                                  epochs,
                                                                  param_combination,
                                                                  trials,
                                                                  best_trial)
                            pbar.update(1)

        checkpoint(self.plot_title, self.save_file_dir, self.save_file_name, trials)

    def __gen_ff_params(self):
        ff_params = [ff_thresh
                     for ff_thresh in np.arange(*self.ff_thresh_params)]
        return list(itertools.product(ff_params, repeat=len(self.layers) - 1))

    def __run_trial(self,
                    train_data,
                    test_data,
                    adam_config,
                    bsize,
                    learn_rate,
                    num_epochs,
                    params,
                    total_trials,
                    top_trial):
        tracker = TrackUsage()
        tracker.start()
        y_true, y_pred, layers = [], [], []
        trial = {'batch_size': bsize,
                 'lr': learn_rate,
                 'epochs': num_epochs,
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
                 'overall_auc': None}

        for i, thresh in enumerate(params):
            trial['ff_thresh_layer_%s' % i] = thresh
            layers.append(FFLinear(adam_config,
                                   self.layers[i],
                                   self.layers[i + 1],
                                   num_epochs=num_epochs,
                                   thresh=thresh))

        try:
            ffnet = FFSupervisedVision(layers, self.total_classes, device=self.device)

            for data, labels in train_data:
                data, labels = data.to(self.device), labels.to(self.device)
                ffnet.train(data, labels)

            for i, loss in enumerate(ffnet.final_losses):
                trial['ff_loss_layer_%s' % i] = loss

            for data, labels in test_data:
                data = data.to(self.device)
                y_pred.extend(ffnet.predict(data).tolist())
                y_true.extend(labels.tolist())

            trial['accuracy'] = accuracy_score(y_true, y_pred)
            trial['f1_score'] = f1_score(y_true, y_pred, average='macro')
            trial['precision'] = precision_score(y_true, y_pred, average='macro')
            trial['recall'] = recall_score(y_true, y_pred, average='macro')

            tracker.stop()
            trial['avg_cpu_perc'] = tracker.avg_cpu_perc
            trial['avg_cpu_memory'] = tracker.avg_cpu_memory
            trial['max_cpu_memory'] = tracker.max_cpu_memory
            trial['cpu_memory_start'] = tracker.cpu_memory_start
            trial['avg_gpu_perc'] = tracker.avg_gpu_perc
            trial['avg_gpu_memory'] = tracker.avg_gpu_memory
            trial['max_gpu_memory'] = tracker.max_gpu_memory
            trial['gpu_memory_start'] = tracker.gpu_memory_start
            trial['time_duration'] = tracker.time_duration
            total_trials.append(trial)

            if top_trial < trial['accuracy']:
                top_trial = trial['accuracy']
                trial['overall_auc'] = checkpoint(self.plot_title,
                                                  self.save_file_dir,
                                                  self.save_file_name,
                                                  total_trials,
                                                  y_true, y_pred)
                print('\nBest Model:', trial)
                torch.save(ffnet.state_dict(),
                           os.path.join(self.save_file_dir,
                                        '{0}_model.pth'.format(self.save_file_name)))
            else:
                trial['overall_auc'] = checkpoint(self.plot_title,
                                                  self.save_file_dir,
                                                  self.save_file_name,
                                                  total_trials)

        except Exception as e:
            print('Failed Trial:', trial, '-', e)
            format_exc()
        finally:
            del ffnet, layers, y_true, y_pred, tracker

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            gc.collect()

        return total_trials, top_trial


def checkpoint(title: str,
               file_dir: str,
               file_name: str,
               total_trials: list,
               best_y_true: list = [],
               best_y_pred: list = []):
    try:
        overall_auc = None
        trials_df = pd.DataFrame(total_trials)
        trials_df.to_csv(os.path.join(file_dir,
                                      '{0}_trials.csv'.format(file_name)), index=False)

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
                plt.title(f'{title} ROC Curve - Class {i}')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(file_dir,
                                         '{0}_roc_curve_class_{1}.png'.format(file_name, i)))
                plt.close()
                del fpr, tpr, roc_auc

            overall_auc = roc_auc_score(bin_y_true, bin_y_pred, multi_class="ovr")

            conf_matrix = confusion_matrix(best_y_true, best_y_pred)
            cm_df = pd.DataFrame(conf_matrix, index=[i for i in uniq_labels],
                                 columns=[i for i in uniq_labels])

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt='d')
            plt.title(f'{title} Confusion Matrix')
            plt.ylabel('Actual Labels')
            plt.xlabel('Predicted Labels')
            plt.savefig(os.path.join(file_dir,
                                     '{0}_confusion_matrix.png'.format(file_name)))
            plt.close()

            del cm_df, conf_matrix, bin_y_pred, bin_y_true, uniq_labels
    except Exception as e:
        print('Failed Recording Checkpoint -', e)
        format_exc()
    finally:
        del trials_df
        plt.close('all')
        gc.collect()

    return overall_auc
