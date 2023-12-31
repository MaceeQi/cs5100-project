import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
        '''
        for i in range(100):
            loss = logistic
            if prev_loss - loss < 0.001:
                break
                
            update of weights
        '''

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



if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 50

    # Parameters for batching
    max_iters = [3000, 5000, 6000, 7000]
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    # solvers = ['lbfgs', 'liblinear', 'newton-cg']
    batch_sizes = [100, 200, 400, 600]

    for max_iter in max_iters:
        for solver in solvers:
            for batch_size in batch_sizes:

                # Retrieve dataloaders (train and test)
                # Dataloader contains batches of dataset. A batch contains: [[2D array of image * batch size], [image labels * batch size]]
                train, test = create_mnist_dataset(batch_size=batch_size)

                # Create instance of logistic regression model
                model = LogisticRegression(solver=solver, random_state=random_seed, max_iter=max_iter)

                # Train logistic regression model batch by batch
                trained_model = train_model(model, train)

                # Test model on test dataset - keep track of true and predicted labels
                true_labels, predicted_labels = test_model(trained_model, test)

                # Determine accuracy metric to evaluate the logistic regression model
                accuracy = accuracy_score(true_labels, predicted_labels)
                print("Max iterations: " + str(max_iter))
                print("Solver: " + solver)
                print("Batch size: " + str(batch_size))
                print("Accuracy: " + str(accuracy))
                print("------------------------------------------------")

