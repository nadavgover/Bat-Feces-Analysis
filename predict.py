import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
from torch.autograd import Variable
import numpy as np

from cnn import CNN, compose
from common import read_data
from fruit_label_enum import create_fruit_labels


def predict(model, data_file, transform=None, fruit_label_enum=create_fruit_labels(fruits=("apple", "banana", "mix")),
            data_width=2100, confidence_threshold=0.7):

    data_to_predict = __get_data_to_predict(data_file=data_file, data_width=data_width)

    # preparation for moving to predicting from a directory and not a single file
    amount_of_data = data_to_predict.shape[0]
    # convert string representation of labels to int
    # for fruit in FruitLabel:
    #     correct_labels = np.where(correct_labels == fruit.name, fruit.value, correct_labels)
    # correct_labels = torch.from_numpy(correct_labels.astype(np.longlong))
    # fruit_label_enum = create_fruit_labels(fruits=fruits)

    if transform:
        data_to_predict = np.reshape(data_to_predict, (-1, 1))
        data_to_predict = transform(data_to_predict).reshape(amount_of_data, 1, 2, -1)
    else:
        data_to_predict = torch.from_numpy(data_to_predict.reshape(amount_of_data, 1, 2, -1))

    # total = 0.0
    # correct = 0.0
    for spectrum in data_to_predict:
        # if transform:
        #     spectrum = transform(spectrum).reshape(1, 1, 2, -1)
        # else:
        spectrum = spectrum.view(1, 1, 2, -1)

        # this line also works (not as good), but don't do the transform above
        # spectrum = torch.from_numpy(spectrum.reshape(1, 1, 2, -1))

        # Run the spectrum through the model
        outputs = model(Variable(spectrum.float()))

        # Brings us probabilities
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        # Get prediction and the confidence (probability) by taking the maximal value of the outputs
        confidence, prediction = torch.max(outputs.data, 1)
        if confidence < confidence_threshold:
            prediction = "Unknown"
        else:  # convert enum value to name
            prediction = fruit_label_enum(prediction.item()).name.capitalize()

        return confidence.item(), prediction


def __get_data_to_predict(data_file, data_width):
    data_read = read_data(filename=data_file)
    data_numpy = np.vstack((np.array(data_read["x"]), np.array(data_read["y"])))

    # make all the data the same size, clip the end of it. The end is not interesting anyway
    data_numpy = data_numpy[:, 0:data_width]
    return data_numpy.reshape((1, 2, data_width))


def load_model(path_to_model, **model_kwargs):
    model = CNN(**model_kwargs)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

if __name__ == '__main__':
    # Hyper parameters
    BATCH_SIZE = 20
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    KERNEL_SIZE = (1, 10)
    PADDING = (1, 5)
    DROP_PROB = 0.2
    BATCH_NORMALIZATION = True
    DROPOUT = True

    # Path of pre-trained model
    MODEL_PATH = 'num_epochs_{}_lr_{}_batch_size_{}_drop_prob_{}.pth' \
                 .format(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DROP_PROB)

    # Get the dataset
    transform = compose(transforms.ToTensor(), minmax_scale)
    # train_data_loader = DataLoader("train", batch_size=BATCH_SIZE, transform=transform)
    # test_data_loader = DataLoader("test", batch_size=BATCH_SIZE, transform=transform)

    # load the pre-trained model
    model = load_model(MODEL_PATH, batch_normalization=BATCH_NORMALIZATION, dropout=DROPOUT, drop_prob=DROP_PROB,
                       kernel_size=KERNEL_SIZE, padding=PADDING)

    predict(model=model, data_file="apple neg.txt", transform=transform,
            fruit_label_enum=create_fruit_labels(fruits=("apple", "banana", "mix")),
            data_width=2100, confidence_threshold=0.7)
