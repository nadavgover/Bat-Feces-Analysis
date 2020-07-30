from pathlib import Path
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale
import copy
import numpy as np
import math
import statistics as stat

from validate_data import get_valid_and_invalid_files
from create_data import create_dataset, train_test_split, save_to_file
from data_loader import DataLoader
from cnn import compose, train_model, CNN
from knn import KNN
import plot_data
from fruit_label_enum import create_fruit_labels
from predict import load_model, predict


def main(train_spectrum_path=r"dataset/train_spectrum.npy", test_spectrum_path=r"dataset/test_spectrum.npy",
         train_labels_path=r"dataset/train_labels.npy", test_labels_path=r"dataset/test_labels.npy",
         batch_size=20, learning_rate=0.01, num_epochs=20, kernel_size=(1, 10), padding=(1, 5), dropout=True,
         drop_prob=0.2, batch_normalization=True, weight_decay=True, weight_decay_amount=0.01, data_width=2100,
         model_save_path=r"model.pth", fruits=("apple", "banana", "mix"), create_dataset_now=False, root_dir="YOMIRAN",
         num_channels_layer1=3, num_channels_layer2=6, sample_time="after 5", sample_location="anal", sample_type="pos",
         tolerance=5, number_of_samples_to_alter=100, size_of_dataset=60000, train_data_percentage=0.8, train_now=False,
         show_statistics=True, predict_now=False, file_to_predict=r"apple neg.txt", confidence_threshold=0.7,
         validate_hierarchy=True, validate_filename_format=True, validate_empty_file=True,
         create_dataset_progress_bar_intvar=None, fc1_amount_output_nodes=1000,
         fc2_amount_output_nodes=500, fc3_amount_output_node=100, stretch_data=True, knn=False,
         cross_validation_iterations=5):

    # create data set
    if create_dataset_now:
        valid_files, _ = get_valid_and_invalid_files(root_dir=root_dir, validate_empty_file=validate_empty_file,
                                                     validate_filename_format=validate_filename_format,
                                                     validate_hierarchy=validate_hierarchy)
        create_dataset(data_files=valid_files, fruits=fruits, size_of_dataset=size_of_dataset,
                       train_data_percentage=train_data_percentage, tolerance=tolerance,
                       number_of_samples_to_alter=number_of_samples_to_alter,
                       train_spectrum_path=Path(train_spectrum_path), train_labels_path=Path(train_labels_path),
                       test_spectrum_path=Path(test_spectrum_path), test_labels_path=Path(test_labels_path),
                       data_width=data_width, sample_time=sample_time, sample_location=sample_location,
                       create_dataset_progress_bar_intvar=create_dataset_progress_bar_intvar, stretch_data=stretch_data,
                       sample_type=sample_type)

    # transformation of dataset
    transform = compose(transforms.ToTensor(), minmax_scale)
    # get the labels enum
    if fruits:
        fruit_label_enum = create_fruit_labels(fruits=fruits)
    # transform = transforms.ToTensor()

    if train_now:
        # Get the dataset
        train_data_loader = DataLoader("train", train_spectrum_path=train_spectrum_path,
                                       train_labels_path=train_labels_path,
                                       batch_size=batch_size, transform=transform)
        test_data_loader = DataLoader("test", test_spectrum_path=test_spectrum_path, test_labels_path=test_labels_path,
                                      batch_size=batch_size, transform=transform)

        if knn:
            train_data_loader_size_calculator = copy.deepcopy(train_data_loader)
            amount_train_data = 0
            for spectrum, labels in train_data_loader_size_calculator.load_data():
                amount_train_data += spectrum.shape[0]

            for k in range(1, amount_train_data):
                model = KNN(k=k, train_data_loader=train_data_loader, test_data_loader=test_data_loader)
                accuracy = model.train()
                print("k={}\tAccuracy: {:.3f}%".format(k, accuracy * 100))

        else:
            train_data_loader_size_calculator = copy.deepcopy(train_data_loader)
            amount_train_data = 0
            fruits_from_dataset = []
            for spectrum, labels in train_data_loader_size_calculator.load_data():
                amount_train_data += spectrum.shape[0]

                # deprecated
                if fruits is None:
                    for label in labels:
                        if label not in fruits_from_dataset:
                            fruits_from_dataset.append(label)

            # deprecated
            if fruits is None:
                fruit_label_enum = create_fruit_labels(fruits=fruits_from_dataset)
                fruits = fruits_from_dataset

            cross_validation_losses = []
            cross_validation_accuracies_train = []
            cross_validation_accuracies_test = []
            cross_validation_true_labels = []
            cross_validation_predictions_of_last_epoch = []
            statistics_of_all_iterations = []
            for i in range(cross_validation_iterations):
                print("Cross validation iteration: {}/{}".format(i+1, cross_validation_iterations))
                # Get the dataset
                train_data_loader = DataLoader("train", train_spectrum_path=train_spectrum_path,
                                               train_labels_path=train_labels_path,
                                               batch_size=batch_size, transform=transform)
                test_data_loader = DataLoader("test", test_spectrum_path=test_spectrum_path,
                                              test_labels_path=test_labels_path,
                                              batch_size=batch_size, transform=transform)

                # initialize the neural net
                model = CNN(amount_of_labels=len(fruit_label_enum), batch_normalization=batch_normalization,
                            dropout=dropout, drop_prob=drop_prob, kernel_size=kernel_size, padding=padding,
                            data_width=data_width, data_height=1, num_channels_layer1=num_channels_layer1,
                            num_channels_layer2=num_channels_layer2, fc1_amount_output_nodes=fc1_amount_output_nodes,
                            fc2_amount_output_nodes=fc2_amount_output_nodes, fc3_amount_output_node=fc3_amount_output_node)

                # train the model
                statistics = train_model(model=model, fruit_label_enum=fruit_label_enum, train_data_loader=train_data_loader,
                                         test_data_loader=test_data_loader, num_epochs=num_epochs,
                                         learning_rate=learning_rate, batch_size=batch_size, weight_decay=weight_decay,
                                         weight_decay_amount=weight_decay_amount, model_save_path=model_save_path,
                                         train_dataset_size=amount_train_data)

                statistics_of_all_iterations.append(statistics)
                losses, accuracies_train, accuracies_test, true_labels, predictions_of_last_epoch = statistics
                cross_validation_losses.extend([losses[-1]])
                cross_validation_accuracies_train.extend([accuracies_train[-1]])
                cross_validation_accuracies_test.extend([accuracies_test[-1]])
                cross_validation_true_labels.extend(true_labels),
                cross_validation_predictions_of_last_epoch.extend(list(predictions_of_last_epoch))

                # shuffle the data for cross validation
                existing_data = None
                existing_labels = None
                train_data_loader_copy = copy.deepcopy(train_data_loader)
                test_data_loader_copy = copy.deepcopy(test_data_loader)
                amount_train_data = 0
                amount_test_data = 0
                for spectrum, labels in train_data_loader_copy.load_data():
                    amount_train_data += spectrum.shape[0]
                    if existing_data is None:
                        existing_data = spectrum
                        existing_labels = labels
                    else:
                        existing_data = np.vstack((existing_data, spectrum))
                        existing_labels = np.hstack((existing_labels, labels))

                for spectrum, labels in test_data_loader_copy.load_data():
                    amount_test_data += spectrum.shape[0]
                    if existing_data is None:
                        existing_data = spectrum
                        existing_labels = labels
                    else:
                        existing_data = np.vstack((existing_data, spectrum))
                        existing_labels = np.hstack((existing_labels, labels))

                amount_of_data = amount_train_data + amount_test_data
                existing_data = existing_data.reshape((amount_of_data, data_width))

                train_data_percentage = math.ceil(10 * (amount_train_data / amount_of_data)) / 10
                train_data, train_labels, test_data, test_labels = train_test_split(existing_data=existing_data,
                                                                                    existing_labels=existing_labels,
                                                                                    train_data_percentage=train_data_percentage)

                # saving the data to the corresponding file
                save_to_file(file=Path(train_spectrum_path), data_to_save=train_data, mode="wb")
                save_to_file(file=Path(train_labels_path), data_to_save=train_labels, mode="wb")
                save_to_file(file=Path(test_spectrum_path), data_to_save=test_data, mode="wb")
                save_to_file(file=Path(test_labels_path), data_to_save=test_labels, mode="wb")

            accuracies_test_mean = stat.mean(cross_validation_accuracies_test)
            accuracies_test_std = stat.stdev(cross_validation_accuracies_test)
            print("Test accuracies mean: {}".format(accuracies_test_mean))
            print("Test accuracies standard deviation: {}".format(accuracies_test_std))
            plot_data.plot_box_plot(test_accuracies=cross_validation_accuracies_test)

            # plot the statistics
            if show_statistics:
                # plot_data.plot_train_statistics(x_values=range(len(losses)), y_values=losses, x_label="Epoch",
                #                                 y_label="Loss")
                # plot_data.plot_train_statistics(x_values=range(len(accuracies_train)), y_values=accuracies_train,
                #                                 x_label="Epoch", y_label="Train accuracy")
                # plot_data.plot_train_statistics(x_values=range(len(accuracies_test)), y_values=accuracies_test,
                #                                 x_label="Epoch", y_label="Test accuracy")

                # statistics = statistics_of_all_iterations[0]
                # losses, accuracies_train, accuracies_test, true_labels, predictions_of_last_epoch = statistics
                # plot_data.plot_train_statistics1(losses=losses, train_accuracy=accuracies_train,
                #                                  test_accuracy=accuracies_test)

                plot_data.plot_confusion_matrix(true_labels=cross_validation_true_labels,
                                                predictions=cross_validation_predictions_of_last_epoch,
                                                fruits=fruits, show_null_values=True, show_plot=True)
                # plot_data.plot_classification_report(true_labels=true_labels, predictions=predictions_of_last_epoch,
                #                                      show_plot=True)

    if predict_now:
        model = load_model(model_save_path, amount_of_labels=len(fruit_label_enum),
                           batch_normalization=batch_normalization, dropout=dropout,
                           drop_prob=drop_prob, kernel_size=kernel_size, padding=padding, data_width=data_width,
                           data_height=1, num_channels_layer1=num_channels_layer1,
                           num_channels_layer2=num_channels_layer2, fc1_amount_output_nodes=fc1_amount_output_nodes,
                           fc2_amount_output_nodes=fc2_amount_output_nodes,
                           fc3_amount_output_node=fc3_amount_output_node)

        confidence, prediction = predict(model=model, data_file=file_to_predict, transform=transform,
                                         fruit_label_enum=fruit_label_enum, data_width=data_width,
                                         confidence_threshold=confidence_threshold)

        return confidence, prediction


if __name__ == '__main__':
    # train_spectrum_path = r"dataset/train_spectrum_apple_banana_original_size.npy"
    # test_spectrum_path = r"dataset/test_spectrum_apple_banana_original_size.npy"
    # train_labels_path = r"dataset/train_labels_apple_banana_original_size.npy"
    # test_labels_path = r"dataset/test_labels_apple_banana_original_size.npy"
    fruits = ["apple", "banana", "mix"]
    # train_spectrum_path = r"dataset/train_spectrum_after5_anal_data_original.npy"
    # test_spectrum_path = r"dataset/test_spectrum_after5_anal_data_original.npy"
    # train_labels_path = r"dataset/train_labels_after5_anal_data_original.npy"
    # test_labels_path = r"dataset/test_labels_after5_anal_data_original.npy"

    # train_spectrum_path = r"dataset/train_spectrum_after5_anal_10000.npy"
    # test_spectrum_path = r"dataset/test_spectrum_after5_anal_10000.npy"
    # train_labels_path = r"dataset/train_labels_after5_anal_10000.npy"
    # test_labels_path = r"dataset/test_labels_after5_anal_10000.npy"

    # train_spectrum_path = r"dataset/train_spectrum_after5_anal_data_original_width900.npy"
    # test_spectrum_path = r"dataset/test_spectrum_after5_anal_data_original_width900.npy"
    # train_labels_path = r"dataset/train_labels_after5_anal_data_original_width900.npy"
    # test_labels_path = r"dataset/test_labels_after5_anal_data_original_width1900.npy"
    train_spectrum_path = r"dataset/train_spectrum.npy"
    test_spectrum_path = r"dataset/test_spectrum.npy"
    train_labels_path = r"dataset/train_labels.npy"
    test_labels_path = r"dataset/test_labels.npy"

    main(create_dataset_now=True, num_epochs=100, kernel_size=(1, 2), padding=(0, 0), size_of_dataset=5000,
         model_save_path=r"trained_models/model_dont_use.pth",
         batch_size=1, train_now=True, predict_now=False, file_to_predict="banana neg.txt",
         train_spectrum_path=train_spectrum_path, test_spectrum_path=test_spectrum_path,
         train_labels_path=train_labels_path, test_labels_path=test_labels_path, show_statistics=True,
         stretch_data=False, sample_location="anal", sample_time="after 5", fruits=fruits, data_width=2100,
         num_channels_layer1=3, num_channels_layer2=6, fc1_amount_output_nodes=200, fc2_amount_output_nodes=500,
         fc3_amount_output_node=100, tolerance=100, number_of_samples_to_alter=250, knn=False, sample_type="pos",
         cross_validation_iterations=5)
