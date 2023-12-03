import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def find_cifar100_mean_std(batch_size: int = 1000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = CIFAR100(root='./data', train=True, download=True, transform=transform)
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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((*mean,), (*std,)),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    train_data = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='./data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=train_batch_size, shuffle=True),\
           DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


# Helper function to flatten dataset images from 2D array to 1D array
def flatten_images(images):
    # Get batch size
    batch_size = images.size(0)

    # Reshape tensor, but preserve batch size
    flattened = images.view(batch_size, -1)

    # Return 1D numpy array of flattened batch of images
    return flattened.numpy()


# Trains the model on data in the training dataset
def train_model(model, train_dataloader):
    # Train model batch by batch
    for batch in train_dataloader:
        images = batch[0]
        labels = batch[1]

        # Flatten images from 2D array to 1D array
        train_images = flatten_images(images)

        # Convert labels to numpy array
        train_labels = labels.numpy()

        # Train the model on the current batch's data
        model.fit(train_images, train_labels)

    # Return the model that has been trained on all batches
    return model


# Tests the trained model on the test dataset
def test_model(model, test_dataloader):
    # Keep track of true and predicted labels
    true_labels, predicted_labels = [], []

    # Go through each batch in test dataloader
    for batch in test_dataloader:
        images = batch[0]
        labels = batch[1]

        # Flatten images from 2D array to 1D array
        test_images = flatten_images(images)

        # Convert labels to numpy array and add to true labels array to keep track
        test_labels = labels.numpy()
        true_labels.extend(test_labels)

        # Use trained model to predict labels and add predictions to predicted labels array to keep track
        predictions = model.predict(test_images)
        predicted_labels.extend(predictions)

    return true_labels, predicted_labels


def track_results(dataset, max_iter, solver, regularization, accuracy):
    file_name = (dataset + "_lr_results.txt")

    with open(file_name, 'a') as file:
        file.write("Max iterations: " + str(max_iter) + ", Solver: " + solver + ", Regularization: " + str(regularization)
                   + ", Accuracy: " + str(accuracy) + "\n")


def evaluate_performance(dataset, path_dir, true_labels, predicted_labels):
    # Performance metrics (accuracy, precision, recall, F1-score, confusion matrix)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    file_name = (dataset + "_lr_best_model.txt")
    with open(file_name, 'a') as file:
        file.write("Accuracy: " + str(accuracy) + ", Precision: " + str(precision)
                   + ", Recall: " + str(recall) + ", F1 score: " + str(f1) + "\n")

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


if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 50

    solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
    regularization = [0.001, 0.01, 0.1, 1, 10]
    max_iters = [5000]

    # Retrieve dataloaders (train and test)
    # Dataloader contains batches of dataset. A batch contains: [[2D array of image * batch size], [image labels * batch size]]
    train, test = create_cifar100_dataset(train_batch_size=50000, test_batch_size=10000)

    best_accuracy = float('-inf')
    best_preds = []
    best_actuals = []
    for solver in solvers:
        for c in regularization:
            for max_iter in max_iters:
                # Create instance of logistic regression model
                model = LogisticRegression(C=c, random_state=random_seed, max_iter=max_iter, solver=solver)

                # Train logistic regression model batch by batch
                trained_model = train_model(model, train)

                # Test model on test dataset - keep track of true and predicted labels
                true_labels, predicted_labels = test_model(trained_model, test)

                # Determine accuracy metric to evaluate the logistic regression model
                accuracy = accuracy_score(true_labels, predicted_labels)
                track_results("cifar100", max_iter, solver, c, accuracy)

                # Keep track of best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_preds = predicted_labels
                    best_actuals = true_labels

                    print("Best Model:")
                    print("Regularization: " + str(c))
                    print("Max iterations: " + str(max_iter))
                    print("Solver: " + solver)
                    print("Accuracy: " + str(accuracy))
                    print("------------------------------------------------")

    # Evaluate best model performance (accuracy, precision, recall, F1-score, confusion matrix)
    path_dir = "/Users/maceeqi/Desktop/MSCS/7_Fall 2023_5100/Foundations of AI/Project/cs5100-project/cifar100"
    evaluate_performance("CIFAR-100", path_dir, best_actuals, best_preds)
