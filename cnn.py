import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
from torch.autograd import Variable
import numpy as np
# import enum
import functools
import time
# import itertools

from data_loader import DataLoader
from fruit_label_enum import create_fruit_labels


def compose(*functions):
    """Function composition g(f(x))"""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# FRUITS = ["apple", "banana", "mix"]
# FruitLabel = enum.Enum('FruitLabel', zip(FRUITS, itertools.count()))


# class FruitLabel(enum.Enum):
#     """Labels enum"""
#     apple = 0
#     banana = 1
#     mix = 2

TRAIN_DATASET_SIZE = 48000


def calculate_output_shape(layer_name, h_in, w_in, kernel_size=(1, 10), padding=0, stride=None, dilation=(1, 1)):
    if layer_name not in ["conv", "pool"]:
        raise ValueError("layer_name must be in ['conv', 'pool']")

    if type(padding) != tuple:
        padding = (padding, padding)
    if type(kernel_size) != tuple:
        kernel_size = (kernel_size, kernel_size)

    if stride is None and layer_name == "pool":
        stride = kernel_size
    if stride is None and layer_name == "conv":
        stride = 1
    if type(stride) != tuple:
        stride = (stride, stride)

    if type(dilation) != tuple:
        dilation = (dilation, dilation)

    h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / float(stride[0]) + 1)
    w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / float(stride[1]) + 1)
    return h_out, w_out


class CNN(nn.Module):
    def __init__(self, batch_normalization=True, dropout=True, drop_prob=0.3, kernel_size=(1, 10), padding=(1, 5),
                 amount_of_labels=3, data_width=2100, data_height=2, num_channels_layer1=3, num_channels_layer2=6):

        super(CNN, self).__init__()
        if batch_normalization:
            self.layer1 = nn.Sequential(nn.Conv2d(1, num_channels_layer1, kernel_size=kernel_size, padding=padding),
                                        nn.BatchNorm2d(num_channels_layer1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(1, num_channels_layer1, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))
        if dropout:
            self.dropout = nn.Sequential(nn.Dropout(p=drop_prob))
        else:
            self.dropout = lambda x: x  # identity function

        if batch_normalization:
            self.layer2 = nn.Sequential(nn.Conv2d(num_channels_layer1, num_channels_layer2, kernel_size=kernel_size,
                                                  padding=padding),
                                        nn.BatchNorm2d(num_channels_layer2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))
        else:
            self.layer2 = nn.Sequential(nn.Conv2d(num_channels_layer1, num_channels_layer2, kernel_size=kernel_size,
                                                  padding=padding),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))

        # calculate shape of data after layer 1
        data_shape_layer1_after_conv2d = calculate_output_shape(layer_name="conv", h_in=data_height, w_in=data_width,
                                                                kernel_size=kernel_size, padding=padding)
        h_out, w_out = data_shape_layer1_after_conv2d
        data_shape_layer1_after_maxpool = calculate_output_shape(layer_name="pool", h_in=h_out, w_in=w_out,
                                                                 kernel_size=kernel_size)
        h_out, w_out = data_shape_layer1_after_maxpool

        # calculate shape of data after layer 2
        data_shape_layer2_after_conv2d = calculate_output_shape(layer_name="conv", h_in=h_out, w_in=w_out,
                                                                kernel_size=kernel_size, padding=padding)
        h_out, w_out = data_shape_layer2_after_conv2d
        data_shape_layer2_after_maxpool = calculate_output_shape(layer_name="pool", h_in=h_out, w_in=w_out,
                                                                 kernel_size=kernel_size)
        h_out, w_out = data_shape_layer2_after_maxpool

        # fully connected layer
        self.fc1 = nn.Linear(h_out * w_out * num_channels_layer2, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, amount_of_labels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        # out = torch.nn.functional.softmax(out, dim=1)  # this is a test, brings us probabilities
        return out


def get_accuracy(epoch, model, batch_size, data_loader, fruit_label_enum):
    correct = 0.0
    total = 0.0
    i = epoch * batch_size
    for spectrum, labels in data_loader.load_data():
        spectrum = Variable(spectrum.float())
        # convert string representation of labels to int
        for fruit in fruit_label_enum:
            labels = np.where(labels == fruit.name, fruit.value, labels)
        labels = torch.from_numpy(labels.astype(np.longlong))

        outputs = model(spectrum)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if i == (epoch+1) * batch_size:
            break
    return (100 * correct / total).item()


def train_model(model, fruit_label_enum, train_data_loader, test_data_loader, num_epochs=50,
                learning_rate=0.01, batch_size=20, weight_decay=False, weight_decay_amount=0.01,
                model_save_path="model.pth"):
    """Trains a neural network"""

    # specify loss function (categorial cross entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate
    if weight_decay:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay_amount)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    accuracies_train = []
    accuracies_test = []
    print("Started training")
    model.train()  # set parameters for training
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        model.eval()  # set parameters for testing
        accuracies_train.append(get_accuracy(epoch=epoch, data_loader=train_data_loader, model=model,
                                             batch_size=batch_size, fruit_label_enum=fruit_label_enum))
        accuracies_test.append(get_accuracy(epoch=epoch, data_loader=test_data_loader, model=model,
                                            batch_size=batch_size, fruit_label_enum=fruit_label_enum))

        model.train()  # set parameters for training
        for i, (spectrum, labels) in enumerate(train_data_loader.load_data()):
            spectrum = Variable(spectrum.float())
            # converting fruit names to int values
            for fruit in fruit_label_enum:
                labels = np.where(labels == fruit.name, fruit.value, labels)
            labels = torch.from_numpy(labels.astype(np.longlong))

            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # clear the gradients of all optimized parameters
            outputs = model(spectrum)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            train_loss += loss.item() * spectrum.size(0)

        train_loss /= TRAIN_DATASET_SIZE
        elapsed_time = time.time() - start_time
        # Log to screen
        print("Epoch: {}/{} \tTraining loss: {:.6f} \tTrain accuracy: {:.6f}% \tTest accuracy: {:.6f}%\t"
              "Epoch time (minutes): {:.2f}".format(
               epoch + 1, num_epochs, train_loss, accuracies_train[-1], accuracies_test[-1], elapsed_time / 60))

    # plot_train_statistics(x_values=range(len(losses)), y_values=losses, x_label="Iteration", y_label="Loss")
    # plot_train_statistics(x_values=range(len(accuracies_train)), y_values=accuracies_train,
    #            x_label="Epoch", y_label="Train accuracy")
    # plot_train_statistics(x_values=range(len(accuracies_test)), y_values=accuracies_test,
    #            x_label="Epoch", y_label="Train accuracy", show_plot=True)

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    # return the statistics
    return losses, accuracies_train, accuracies_test


def plot_train_statistics(x_values, y_values, x_label, y_label, show_plot=False):
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    # ax.legend(legend_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_plot:
        plt.show()

if __name__ == '__main__':
    # Hyper parameters
    BATCH_SIZE = 20
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    KERNEL_SIZE = (1, 10)
    PADDING = (1, 5)
    DROP_PROB = 0.2
    MODEL_SAVE_PATH = 'num_epochs_{}_lr_{}_batch_size_{}_drop_prob_{}.pth'\
        .format(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DROP_PROB)

    # Get the dataset
    train_spectrum_path = r"dataset/train_spectrum.npy"
    test_spectrum_path = r"dataset/test_spectrum.npy"
    train_labels_path = r"dataset/train_labels.npy"
    test_labels_path = r"dataset/test_labels.npy"
    transform = compose(transforms.ToTensor(), minmax_scale)
    train_data_loader = DataLoader("train", train_spectrum_path=train_spectrum_path,
                                   train_labels_path=train_labels_path,
                                   batch_size=BATCH_SIZE, transform=transform)
    test_data_loader = DataLoader("test", test_spectrum_path=test_spectrum_path, test_labels_path=test_labels_path,
                                  batch_size=BATCH_SIZE, transform=transform)

    # Get the dataset
    # transform = compose(transforms.ToTensor(), minmax_scale)
    # train_data_loader = DataLoader("train", batch_size=BATCH_SIZE, transform=transform)
    # test_data_loader = DataLoader("test", batch_size=BATCH_SIZE, transform=transform)

    # train
    statistics = train_model(train_data_loader=train_data_loader, test_data_loader=test_data_loader,
                             num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                             weight_decay=True, model_save_path=MODEL_SAVE_PATH,
                             fruit_label_enum=create_fruit_labels())
