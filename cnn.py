import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
from torch.autograd import Variable
import numpy as np
import functools
import time


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
                 amount_of_labels=3, data_width=2100, data_height=2, num_channels_layer1=3, num_channels_layer2=6,
                 fc1_amount_output_nodes=1000, fc2_amount_output_nodes=500, fc3_amount_output_node=100):

        super(CNN, self).__init__()

        # layer 1
        if batch_normalization:
            self.layer1 = nn.Sequential(nn.Conv2d(1, num_channels_layer1, kernel_size=kernel_size, padding=padding),
                                        nn.BatchNorm2d(num_channels_layer1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(1, num_channels_layer1, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size))

        # layer 2
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

        # dropout
        if dropout:
            self.dropout = nn.Sequential(nn.Dropout(p=drop_prob))
        else:
            self.dropout = lambda x: x  # identity function

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
        self.fc1 = nn.Linear(w_out * num_channels_layer2, fc1_amount_output_nodes)
        self.fc2 = nn.Linear(fc1_amount_output_nodes, fc2_amount_output_nodes)
        self.fc3 = nn.Linear(fc2_amount_output_nodes, fc3_amount_output_node)
        self.fc4 = nn.Linear(fc3_amount_output_node, amount_of_labels)

        # # paper architecture
        # self.paper_conv1 = nn.Sequential(nn.Conv2d(1, num_channels_layer1, kernel_size=kernel_size, padding=padding),
        #                                  nn.BatchNorm2d(num_channels_layer1),
        #                                  nn.ReLU())
        #
        # self.paper_conv2 = nn.Sequential(nn.Conv2d(num_channels_layer1, num_channels_layer2, kernel_size=kernel_size,
        #                                  padding=padding),
        #                                  nn.BatchNorm2d(num_channels_layer2),
        #                                  nn.ReLU(),
        #                                  nn.MaxPool2d(kernel_size),
        #                                  nn.Dropout(p=0.25))
        #
        # data_shape_layer1_after_conv2d = calculate_output_shape(layer_name="conv", h_in=data_height, w_in=data_width,
        #                                                         kernel_size=kernel_size, padding=padding)
        # h_out, w_out = data_shape_layer1_after_conv2d
        #
        # # calculate shape of data after layer 2
        # data_shape_layer2_after_conv2d = calculate_output_shape(layer_name="conv", h_in=h_out, w_in=w_out,
        #                                                         kernel_size=kernel_size, padding=padding)
        # h_out, w_out = data_shape_layer2_after_conv2d
        # data_shape_layer2_after_maxpool = calculate_output_shape(layer_name="pool", h_in=h_out, w_in=w_out,
        #                                                          kernel_size=kernel_size)
        # h_out, w_out = data_shape_layer2_after_maxpool
        #
        # self.paper_fc1 = nn.Sequential(nn.Linear(h_out * w_out * num_channels_layer2, fc1_amount_output_nodes),
        #                                nn.ReLU(),
        #                                nn.Dropout(p=0.5))
        # self.paper_fc2 = nn.Linear(fc1_amount_output_nodes, amount_of_labels)

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
        return out

    # def forward(self, x):
    #     # like the paper
    #     out = self.paper_conv1(x)
    #     out = self.paper_conv2(out)
    #     out = out.view(out.size(0), -1)
    #     out = self.paper_fc1(out)
    #     out = self.paper_fc2(out)
    #     return out


def get_accuracy_labels_vs_predicted(epoch, model, batch_size, data_loader, fruit_label_enum,
                                     get_labels_predictions=False):
    correct = 0.0
    total = 0.0
    all_true_labels = None
    all_predictions = None
    for spectrum, labels_np_string in data_loader.load_data():
        spectrum = Variable(spectrum.float())
        # convert string representation of labels to int
        labels = np.array([fruit_label_enum[label].value for label in labels_np_string])
        # for fruit in fruit_label_enum:
        #     labels = np.where(labels_np_string == fruit.name, fruit.value, labels)
        labels = torch.from_numpy(labels.astype(np.longlong))

        outputs = model(spectrum)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
        if get_labels_predictions:
            if all_true_labels is None:
                all_true_labels = labels_np_string
            else:
                all_true_labels = np.hstack((all_true_labels, labels_np_string))

            predicted_np = predicted.numpy()
            predicted_np_string = [fruit_label_enum(prediction).name for prediction in predicted_np]
            if all_predictions is None:
                all_predictions = predicted_np_string
            else:
                all_predictions = np.hstack((all_predictions, predicted_np_string))
    accuracy = (100 * correct / total).item()

    return accuracy, all_true_labels, all_predictions


def train_model(model, fruit_label_enum, train_data_loader, test_data_loader, num_epochs=50,
                learning_rate=0.01, batch_size=20, weight_decay=False, weight_decay_amount=0.01,
                model_save_path="model.pth", train_dataset_size=60000):
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
        accuracies_train.append(get_accuracy_labels_vs_predicted(epoch=epoch, data_loader=train_data_loader, model=model,
                                                                 batch_size=batch_size,
                                                                 fruit_label_enum=fruit_label_enum)[0])
        accuracies_test.append(get_accuracy_labels_vs_predicted(epoch=epoch, data_loader=test_data_loader, model=model,
                                                                batch_size=batch_size,
                                                                fruit_label_enum=fruit_label_enum)[0])

        if epoch == num_epochs - 1:
            _, true_labels, predictions_of_last_epoch = get_accuracy_labels_vs_predicted(epoch=epoch,
                                                                                         data_loader=test_data_loader,
                                                                                         model=model,
                                                                                         batch_size=batch_size,
                                                                                         fruit_label_enum=fruit_label_enum,
                                                                                         get_labels_predictions=True)

        model.train()  # set parameters for training
        for spectrum, labels in train_data_loader.load_data():
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
            # losses.append(loss.data.item())
            train_loss += loss.item() * spectrum.size(0)

        train_loss /= train_dataset_size
        losses.append(train_loss)
        elapsed_time = time.time() - start_time
        # Log to screen
        print("Epoch: {}/{} \tTraining loss: {:.6f} \tTrain accuracy: {:.6f}% \tTest accuracy: {:.6f}%\t"
              "Epoch time (minutes): {:.2f}".format(
               epoch + 1, num_epochs, train_loss, accuracies_train[-1], accuracies_test[-1], elapsed_time / 60))

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    # return the statistics
    return losses, accuracies_train, accuracies_test, true_labels, predictions_of_last_epoch


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

    # instantiate model
    model = CNN()

    # train
    statistics = train_model(train_data_loader=train_data_loader, test_data_loader=test_data_loader,
                             num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                             weight_decay=True, model_save_path=MODEL_SAVE_PATH,
                             fruit_label_enum=create_fruit_labels(), model=model)
