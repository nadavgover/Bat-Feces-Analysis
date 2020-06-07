import numpy as np
import random
import copy
from pathlib import Path

from common import read_data
from validate_data import get_valid_and_invalid_files

# FRUITS = ["apple", "banana", "mix"]
SAMPLE_TIMES = ["after 5", "after 8", "before", "all"]
SAMPLE_LOCATIONS = ["oral", "anal", "all"]
MAX_INTERESTING_MOLECULAR_WEIGHT = 750


def create_dataset(data_files, fruits=("apple", "banana", "mix"), sample_time="after 5", sample_location="anal",
                   tolerance=1, number_of_samples_to_alter=100, size_of_dataset=60000, train_data_percentage=0.8,
                   train_spectrum_path=Path("dataset/train_spectrum.npy"), train_labels_path=Path("dataset/train_labels.npy"),
                   test_spectrum_path=Path("dataset/test_spectrum.npy"), test_labels_path=Path("dataset/test_labels.npy"),
                   data_width=2100, stretch_data=False, create_dataset_progress_bar_intvar=None):

    # Get existing data
    existing_data, existing_labels = __get_existing_data(fruits=fruits, data_files=data_files, sample_time=sample_time,
                                                         sample_location=sample_location, data_width=data_width)

    if stretch_data:
        train_dataset_size = int(size_of_dataset * train_data_percentage)
        test_dataset_size = size_of_dataset - train_dataset_size

        # split to train and test
        amount_of_train_data_to_choose = int(train_data_percentage * existing_data.shape[0])
        indices_of_train = random.sample(range(existing_data.shape[0]), amount_of_train_data_to_choose)
        indices_of_test = [i for i in range(existing_data.shape[0]) if i not in indices_of_train]
        # getting the data from the existing data
        train_data = existing_data[indices_of_train]
        train_labels = existing_labels[indices_of_train]
        test_data = existing_data[indices_of_test]
        test_labels = existing_labels[indices_of_test]

        # Create train and test dataset. Existing data will be part of the train dataset
        for type_of_dataset, dataset_size, data, labels in zip(["train", "test"],
                                                               [train_dataset_size, test_dataset_size],
                                                               [train_data, test_data],
                                                               [train_labels, test_labels]):

            __create_train_or_test_dataset(size_of_dataset=dataset_size, type_of_dataset=type_of_dataset,
                                           existing_data=data, existing_labels=labels,
                                           tolerance=tolerance, number_of_samples_to_alter=number_of_samples_to_alter,
                                           train_spectrum_path=train_spectrum_path, train_labels_path=train_labels_path,
                                           test_spectrum_path=test_spectrum_path, test_labels_path=test_labels_path,
                                           data_width=data_width,
                                           create_dataset_progress_bar_intvar=create_dataset_progress_bar_intvar)

    else:  # do not stretch data (do not create more data from existing data)
        train_dataset_size = int(train_data_percentage * existing_data.shape[0])
        indices_of_train = random.sample(range(existing_data.shape[0]), train_dataset_size)
        indices_of_test = [i for i in range(existing_data.shape[0]) if i not in indices_of_train]

        # getting the data from the existing data
        train_data = existing_data[indices_of_train]
        train_labels = existing_labels[indices_of_train]
        test_data = existing_data[indices_of_test]
        test_labels = existing_labels[indices_of_test]

        # saving the data to the corresponding file
        save_to_file(file=train_spectrum_path, data_to_save=train_data, mode="wb")
        save_to_file(file=train_labels_path, data_to_save=train_labels, mode="wb")
        save_to_file(file=test_spectrum_path, data_to_save=test_data, mode="wb")
        save_to_file(file=test_labels_path, data_to_save=test_labels, mode="wb")
    return True


def __create_train_or_test_dataset(size_of_dataset, type_of_dataset, existing_data, existing_labels, tolerance,
                                   number_of_samples_to_alter, train_spectrum_path, train_labels_path,
                                   test_spectrum_path, test_labels_path, data_width,
                                   create_dataset_progress_bar_intvar=None):
    if type_of_dataset not in ["train", "test"]:
        raise ValueError("type_of_dataset must be in ['train', 'test']")
    if size_of_dataset % 1000 != 0:
        raise ValueError("Size of dataset must be a multiple of 1,000\n"
                         "Try changing train_data_percentage and/or size_of_dataset such that:\n"
                         "(train_data_percentage * size_of_dataset) % 1000 = 0")

    # existing data will be part of the training dataset
    for i in range(0, size_of_dataset, 1000):
        if i == 0 and type_of_dataset == "train":
            randomized_data, randomized_labels = __create_random_data_from_existing_data(existing_data=existing_data,
                                                                                         existing_labels=existing_labels,
                                                                                         amount_of_data_to_create=1000-existing_data.shape[0],
                                                                                         tolerance=tolerance,
                                                                                         number_of_samples_to_alter=number_of_samples_to_alter,
                                                                                         data_width=data_width)

            current_data = np.vstack((existing_data, randomized_data))
            current_labels = np.hstack((existing_labels, randomized_labels))
        else:
            randomized_data, randomized_labels = __create_random_data_from_existing_data(existing_data=existing_data,
                                                                                         existing_labels=existing_labels,
                                                                                         amount_of_data_to_create=1000,
                                                                                         tolerance=tolerance,
                                                                                         number_of_samples_to_alter=number_of_samples_to_alter,
                                                                                         data_width=data_width)
            current_data = randomized_data
            current_labels = randomized_labels

        if i == 0:
            writing_to_file_mode = "wb"
        else:
            writing_to_file_mode = "ab"

        if type_of_dataset == "train":
            save_to_file(train_spectrum_path, current_data, writing_to_file_mode)
            save_to_file(train_labels_path, current_labels, writing_to_file_mode)

        else:  # if type_of_dataset == "test"
            save_to_file(test_spectrum_path, current_data, writing_to_file_mode)
            save_to_file(test_labels_path, current_labels, writing_to_file_mode)
        if create_dataset_progress_bar_intvar:
            create_dataset_progress_bar_intvar.set(create_dataset_progress_bar_intvar.get() + 1000)


def save_to_file(file, data_to_save, mode):
    with file.open(mode) as f:
        np.save(f, data_to_save)


def __get_existing_data(data_files, data_width=2100, fruits=("apple", "banana", "mix"), sample_time="after 5",
                        sample_location="anal"):
    if type(sample_time) == list:
        for _sample_time in sample_time:
            if _sample_time.lower() not in SAMPLE_TIMES:
                raise ValueError("sample_time must be in {}".format(SAMPLE_TIMES))
    if type(sample_location) == list:
        for _sample_location in sample_location:
            if _sample_location.lower() not in SAMPLE_LOCATIONS:
                raise ValueError("sample_location must be in {}".format(SAMPLE_LOCATIONS))

    if type(sample_time) == str and sample_time.lower() not in SAMPLE_TIMES:
        raise ValueError("sample_time must be in {}".format(SAMPLE_TIMES))
    if type(sample_location) == str and sample_location.lower() not in SAMPLE_LOCATIONS:
        raise ValueError("sample_location must be in {}".format(SAMPLE_LOCATIONS))

    existing_data = None
    existing_labels = None
    amount_of_data = 0

    # get the data from the valid data files
    for data_file in data_files:
        if type(sample_time) == list:
            sample_time_in_data_file = False
            for _sample_time in sample_time:
                if _sample_time.lower() in data_file.lower():
                    sample_time_in_data_file = True
                    break
            if not sample_time_in_data_file:
                continue
        else:
            if sample_time.lower() != "all":
                if sample_time.lower() not in data_file.lower():
                    continue

        if type(sample_location) == list:
            sample_location_in_data_file = False
            for _sample_location in sample_location:
                if _sample_location.lower() in data_file.lower():
                    sample_location_in_data_file = True
                    break
            if not sample_location_in_data_file:
                continue
        else:
            if sample_location.lower() != "all":
                if sample_location.lower() not in data_file.lower():
                    continue

        label = __get_label(file_path=data_file, fruits=fruits)
        if label == "Unknown fruit":
            continue

        data_read = read_data(filename=data_file)
        data_numpy = np.vstack((np.array(data_read["x"]), np.array(data_read["y"])))

        # make all the data the same size, clip the end of it. The end is not interesting anyway
        if data_numpy.shape[1] <= data_width:
            continue
        data_numpy = data_numpy[:, 0:data_width]

        if existing_data is None:
            existing_data = data_numpy
            existing_labels = np.array([label])
        else:
            existing_data = np.vstack((existing_data, data_numpy))
            existing_labels = np.hstack((existing_labels, label))

        amount_of_data += 1

    existing_data = existing_data.reshape((amount_of_data, 2, data_width))
    return existing_data, existing_labels


def __create_random_data_from_existing_data(existing_data, existing_labels, amount_of_data_to_create,
                                            tolerance, number_of_samples_to_alter, data_width):
    """Create random data by altering (stretching/shrinking) existing samples by the tolerance of the ms machine.
    Keep balance between labels, meaning have the same amount of data from each label"""
    randomized_data = None
    randomized_labels = None
    unique_labels = list(set(existing_labels))
    label_to_randomize_index = 0
    created_data_amount = 0
    while created_data_amount < amount_of_data_to_create:
        label_of_data_to_randomize = unique_labels[label_to_randomize_index % len(unique_labels)]
        label_to_randomize_index += 1
        index_of_data_to_randomize_from = random.choice(np.where(existing_labels == label_of_data_to_randomize)[0])
        data_to_randomize_from = existing_data[index_of_data_to_randomize_from]
        indices_to_alter = list(set(np.random.choice(MAX_INTERESTING_MOLECULAR_WEIGHT, number_of_samples_to_alter)))
        for index_to_alter in indices_to_alter:
            # pick a random number between -tolerance to tolerance
            alter_amount = -tolerance + (random.random() * (2 * tolerance))
            current_data = copy.deepcopy(data_to_randomize_from)
            if current_data[0][index_to_alter] + alter_amount > 0:
                current_data[0][index_to_alter] += alter_amount
        if randomized_data is None:
            randomized_data = current_data
            randomized_labels = np.array([label_of_data_to_randomize])
        else:
            randomized_data = np.vstack((randomized_data, current_data))
            randomized_labels = np.hstack((randomized_labels, label_of_data_to_randomize))
        created_data_amount += 1
    randomized_data = randomized_data.reshape((created_data_amount, 2, data_width))
    return randomized_data, randomized_labels


def __get_label(file_path, fruits):
    """Get the label of the data from the file name.
    Assuming file directory hierarchy.
    If hierarchy is not expected, just check if the filename contains a fruit name"""
    try:
        root, name, fruit, location, time, filename = r"{}".format(file_path).split("\\")
        if fruit.lower() not in fruits:
            return "Unknown fruit"
        return fruit.lower()
    except ValueError:
        for fruit in fruits:
            if fruit in file_path.lower():
                return fruit

        return "Unknown fruit"  # if we got here so we don't know what is the label


if __name__ == '__main__':
    create_now = True
    fruits = ("apple", "banana")
    if create_now:
        valid_files, _ = get_valid_and_invalid_files(root_dir="YOMIRAN", validate_hierarchy=True,
                                                     validate_filename_format=True, validate_empty_file=True)
        create_dataset(data_files=valid_files, fruits=fruits, size_of_dataset=60000, train_data_percentage=0.8,
                       tolerance=1, number_of_samples_to_alter=100, sample_time="after 5",
                       stretch_data=False, train_spectrum_path=Path("train_spectrum_stam.npy"),
                       train_labels_path=Path("train_labels_stam.npy"),
                       test_spectrum_path=Path("test_spectrum_stam.npy"),
                       test_labels_path=Path("test_labels_stam.npy"))
