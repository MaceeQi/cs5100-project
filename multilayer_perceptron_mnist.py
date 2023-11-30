import os

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm


def find_mnist_mean_std(batch_size: int = 1000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    mean = 0
    std = 0

    for batch, _ in train_loader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    return (mean / len(train_loader.dataset)).item(), (std / len(train_loader.dataset)).item()


def create_mnist_dataset(batch_size: int = 25):
    mean, std = find_mnist_mean_std()

    # Transform object to convert input pixels into tensors and standardize them (normalize data)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,)),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    # Download train and test data from MNIST dataset
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = MNIST(root='./data', train=False, download=True, transform=transform)

    # Return dataloaders for train and test datasets
    return DataLoader(train_data, batch_size=batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Class for the multilayer perceptron
class MultiLayerPerceptronModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super().__init__()

        # Input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # Hidden layer sizes
        self.hidden_layer_sizes = hidden_layer_sizes

        # Layers of mlp (3-5)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_layer_sizes[0]),  # first hidden layer
            torch.nn.ReLU(),    # activation function
            torch.nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1]),  # second hidden layer
            torch.nn.ReLU(),    # activation function
            torch.nn.Linear(self.hidden_layer_sizes[1], self.output_size)  # output layer
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def evaluate_performance(dataset, path_dir, true_labels, predicted_labels):
    # Performance metrics (accuracy, precision, recall, F1-score, confusion matrix)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 score: " + str(f1))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(dataset + ' Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(os.path.join(path_dir,
                             '{0}_confusion_matrix.png'.format(dataset)))
    plt.show()
    plt.close()


def track_results(dataset, input_size, best_model, accuracy):
    file_name = (dataset + ".txt")
    layer_sizes = ("[" + str(input_size) + ", " + str(best_model.get('hidden_layer_sizes')[0])
                   + ", " + str(best_model.get('hidden_layer_sizes')[1]) + "]")
    epochs = str(best_model.get('total_epochs'))
    batch_size = str(best_model.get('batch_size'))
    learning_rate = str(best_model.get('learning_rate'))

    with open(file_name, 'a') as file:
        file.write("Layer sizes: " + layer_sizes + ", Epochs: " + epochs + ", Batch size: " + batch_size
                   + ", Learning rate: " + learning_rate + " Accuracy: " + str(accuracy) + "\n")


def train_mlp(dataset, path_dir, input_size, output_size, hidden_layer_sizes, total_epochs, batch_sizes, learning_rates):
    # Keep track of best model (compare using accuracy, store hyperparameters and predicted/actual labels)
    best_accuracy = float('-inf')
    best_model = {}

    # Trials on different combinations of hyperparameters
    for batch_size in tqdm(batch_sizes, position=0, leave=True, desc="Batch Sizes"):
        # Retrieve MNIST dataloaders (train and test)
        train, test = create_mnist_dataset(batch_size)

        for hidden_layer_size in tqdm(hidden_layer_sizes, position=1, leave=True, desc="Hidden Layer Sizes"):
            for learning_rate in tqdm(learning_rates, position=2, leave=True, desc="Learning Rates"):
                for num_of_epochs in tqdm(total_epochs, position=3, leave=True, desc="All Epochs"):
                    # Create MLP model
                    model = MultiLayerPerceptronModel(input_size=input_size, output_size=output_size,
                                                      hidden_layer_sizes=hidden_layer_size)

                    # Instantiate loss function and optimizer
                    loss_function = F.cross_entropy
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    for epoch in tqdm(range(num_of_epochs), position=4, leave=True, desc="Epoch"):
                        # Train MLP
                        for train_features, train_labels in train:
                            # Flatten images
                            train_features = train_features.view(-1, input_size)

                            # Forward pass - get predicted label
                            predicted_y = model(train_features)

                            # Calculate loss from predicted and actual label
                            loss = loss_function(predicted_y, train_labels)

                            # Clear derivatives
                            optimizer.zero_grad()

                            # Backward propagation
                            loss.backward()

                            # Update trainable parameters (weight, bias)
                            optimizer.step()

                        # Test MLP
                        # Change mode of model to evaluation mode
                        model.eval()

                        # Instantiate lists to hold predicted and actual labels
                        predicted_labels, true_labels = [], []

                        for test_features, test_labels in test:
                            # Flatten images
                            test_features = test_features.view(-1, input_size)

                            # Use trained model to predict labels and add predictions to predicted labels
                            predictions = model(test_features)
                            predicted_y = torch.argmax(predictions, dim=1)
                            predicted_labels.extend(predicted_y.tolist())

                            # Add true labels to actual labels array
                            true_labels.extend(test_labels.tolist())

                        # Get accuracy
                        accuracy = accuracy_score(true_labels, predicted_labels)

                        # Save accuracy, predicted labels, true labels, and hyperparameters if best model so far
                        if accuracy > best_accuracy:

                            torch.save(model.state_dict(),
                                       os.path.join(path_dir, '{0}_model.pth'.format('best_' + dataset)))
                            best_accuracy = accuracy
                            best_model = {'best_predicted_labels': predicted_labels,
                                          'best_true_labels': true_labels,
                                          'hidden_layer_sizes': hidden_layer_size,
                                          'total_epochs': num_of_epochs,
                                          'learning_rate': learning_rate,
                                          'batch_size': batch_size
                                          }
                            track_results(dataset, input_size, best_model, accuracy)


    print("3 layers: [" + str(input_size) + ", " + str(best_model.get('hidden_layer_sizes')[0]) + ", "
          + str(best_model.get('hidden_layer_sizes')[1]) + "]")
    print("Epochs: " + str(best_model.get('total_epochs')))
    print("Batch size: " + str(best_model.get('batch_size')))
    print("Learning rate: " + str(best_model.get('learning_rate')))
    print("_________________________________________")

    # Evaluate performance of best model (accuracy, precision, recall, F1-score, confusion matrix)
    best_true_list = best_model.get('best_true_labels')
    best_predicted_list = best_model.get('best_predicted_labels')
    evaluate_performance(dataset, path_dir, best_true_list, best_predicted_list)


if __name__ == "__main__":
    # MLP hyperparameters
    input_size = 784    # 28x28 pixel image flattened to 784-dimensional vector (MNIST)
    output_size = 10    # 10 classes (numbers 0 through 9) (MNIST)
    batch_sizes = [64, 128, 256, 512, 1024]
    hidden_layer_sizes = [(1000, 500), (1000, 100)]    # total layers = # hidden layers + 1 output layer
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    total_epochs = [5, 10, 20, 50]

    # Set seed for consistent results
    torch.manual_seed(50)

    # Conduct 160 trials on MNIST dataset for MLP
    train_mlp(dataset="MNIST",
              path_dir="C:\cs5100",
              input_size=input_size,
              output_size=output_size,
              hidden_layer_sizes=hidden_layer_sizes,
              total_epochs=total_epochs,
              batch_sizes=batch_sizes,
              learning_rates=learning_rates)






