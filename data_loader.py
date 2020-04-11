import os
from pathlib import Path
import numpy as np
import functools
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as transforms
import collections


class DataLoader(object):
    def __init__(self, data_type, train_spectrum_path=None, test_spectrum_path=None,
                 train_labels_path=None, test_labels_path=None, batch_size=50, transform=None):
        if data_type not in ["train", "test"]:
            raise ValueError("data_type must be in ['train', 'test']")
        # if 1000 % batch_size != 0:
        #     raise ValueError("1,000 must be divisible by batch_size")
        self.data_type = data_type.lower()
        self.train_spectrum_path = train_spectrum_path
        self.test_spectrum_path = test_spectrum_path
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path
        if bool(self.train_spectrum_path) ^ bool(self.train_labels_path):
            raise ValueError("If train_spectrum_path was given so test_labels_path must be given too. And vice versa")
        if bool(self.test_spectrum_path) ^ bool(self.test_labels_path):
            raise ValueError("If test_spectrum_path was given so test_labels_path must be given too. And vice versa")
        if train_spectrum_path is None and test_spectrum_path is None and train_labels_path is None and test_labels_path is None:
            raise ValueError("One of the paths should be given, train or test (spectrum and labels)")

        self.batch_size = batch_size
        self.transform = transform

    def __get_spectrum_path_for_loading(self):
        if self.data_type == "train":
            return Path(self.train_spectrum_path)
        else:
            return Path(self.test_spectrum_path)

    def __get_labels_path_for_loading(self):
        if self.data_type == "train":
            return Path(self.train_labels_path)
        else:
            return Path(self.test_labels_path)

    def __load_data_from_file(self, file_path):
        with file_path.open('rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            while f.tell() < fsz:
                yield np.load(f)

    def load_data(self):
        spectrum_file_path = self.__get_spectrum_path_for_loading()
        labels_file_path = self.__get_labels_path_for_loading()
        spectrum_loader = self.__load_data_from_file(spectrum_file_path)
        labels_loader = self.__load_data_from_file(labels_file_path)
        while True:
            try:
                current_spectrum_chunk = next(spectrum_loader)
                current_labels_chunk = next(labels_loader)
                for i in range(0, current_spectrum_chunk.shape[0], self.batch_size):
                    # the last batch is maybe less than the actual batch size, calculate its size
                    if self.batch_size <= current_spectrum_chunk.shape[0] - i:
                        first_dimension_shape = self.batch_size
                    else:
                        first_dimension_shape = current_spectrum_chunk.shape[0] - i

                    spectrum_batch = current_spectrum_chunk[i: i+self.batch_size, :, :].reshape((first_dimension_shape, 1, 2, -1))
                    labels_batch = current_labels_chunk[i:i+self.batch_size].reshape((first_dimension_shape,))
                    if self.transform:
                        spectrum_batch = np.reshape(spectrum_batch, (-1, 1))
                        yield self.transform(spectrum_batch).reshape(first_dimension_shape, 1, 2, -1), labels_batch
                    else:
                        yield spectrum_batch, labels_batch
            except StopIteration:
                break


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

if __name__ == '__main__':
    batch_size = 10
    # train_spectrum_path = r"dataset/train_spectrum_after5_anal_data_original.npy"
    # test_spectrum_path = r"dataset/test_spectrum_after5_anal_data_original.npy"
    # train_labels_path = r"dataset/train_labels_after5_anal_data_original.npy"
    # test_labels_path = r"dataset/test_labels_after5_anal_data_original.npy"
    train_spectrum_path = "train_spectrum_stam.npy"
    test_spectrum_path = "test_spectrum_stam.npy"
    train_labels_path = "train_labels_stam.npy"
    test_labels_path = "test_labels_stam.npy"

    transform = compose(transforms.ToTensor(), minmax_scale)
    train_data_loader = DataLoader("train", train_spectrum_path=train_spectrum_path, train_labels_path=train_labels_path,
                                   batch_size=batch_size, transform=transform)
    test_data_loader = DataLoader("test", test_spectrum_path=test_spectrum_path, test_labels_path=test_labels_path,
                                  batch_size=batch_size, transform=transform)

    amount_train_data = 0
    i = 0
    train_data_counter = collections.Counter()
    for spectrum, labels in train_data_loader.load_data():
        # do something with spectrum and labels
        train_data_counter.update(labels)
        amount_train_data += spectrum.shape[0]

    amount_test_data = 0
    test_data_counter = collections.Counter()
    for spectrum, labels in test_data_loader.load_data():
        # do something with spectrum and labels
        test_data_counter.update(labels)
        amount_test_data += spectrum.shape[0]

    print("Amount of train data: {}".format(amount_train_data))
    print("Amount of test data: {}".format(amount_test_data))
    print("\nTrain dataset contains:")
    for fruit in train_data_counter:
        print(fruit, train_data_counter[fruit])
    print("\nTest dataset contains:")
    for fruit in test_data_counter:
        print(fruit, test_data_counter[fruit])


