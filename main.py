from pathlib import Path
import torchvision.transforms as transforms
from sklearn.preprocessing import minmax_scale

from validate_data import get_valid_and_invalid_files
from create_data import create_dataset
from data_loader import DataLoader
from cnn import compose, train_model, CNN, plot_train_statistics
from fruit_label_enum import create_fruit_labels
from predict import load_model, predict


def main(train_spectrum_path=r"dataset/train_spectrum.npy", test_spectrum_path=r"dataset/test_spectrum.npy",
         train_labels_path=r"dataset/train_labels.npy", test_labels_path=r"dataset/test_labels.npy",
         batch_size=20, learning_rate=0.01, num_epochs=20, kernel_size=(1, 10), padding=(1, 5), dropout=True,
         drop_prob=0.2, batch_normalization=True, weight_decay=True, weight_decay_amount=0.01, data_width=2100,
         model_save_path=r"model.pth", fruits=("apple", "banana", "mix"), create_dataset_now=False, root_dir="YOMIRAN",
         num_channels_layer1=3, num_channels_layer2=6, sample_time="after 5", sample_location="anal", tolerance=1,
         number_of_samples_to_alter=100, size_of_dataset=60000, train_data_percentage=0.8, train_now=False,
         show_statistics=True, predict_now=False, file_to_predict=r"apple neg.txt", confidence_threshold=0.7,
         validate_hierarchy=True, validate_filename_format=True, validate_empty_file=True):

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
                       data_width=data_width, sample_time=sample_time, sample_location=sample_location)

    # transformation of dataset
    transform = compose(transforms.ToTensor(), minmax_scale)
    # get the labels enum
    fruit_label_enum = create_fruit_labels(fruits=fruits)

    if train_now:
        # Get the dataset
        train_data_loader = DataLoader("train", train_spectrum_path=train_spectrum_path,
                                       train_labels_path=train_labels_path,
                                       batch_size=batch_size, transform=transform)
        test_data_loader = DataLoader("test", test_spectrum_path=test_spectrum_path, test_labels_path=test_labels_path,
                                      batch_size=batch_size, transform=transform)

        # initialize the neural net
        model = CNN(amount_of_labels=len(fruit_label_enum), batch_normalization=batch_normalization, dropout=dropout,
                    drop_prob=drop_prob, kernel_size=kernel_size, padding=padding, data_width=data_width, data_height=2,
                    num_channels_layer1=num_channels_layer1, num_channels_layer2=num_channels_layer2)

        # train the model
        statistics = train_model(model=model, fruit_label_enum=fruit_label_enum, train_data_loader=train_data_loader,
                                 test_data_loader=test_data_loader, num_epochs=num_epochs, learning_rate=learning_rate,
                                 batch_size=batch_size, weight_decay=weight_decay,
                                 weight_decay_amount=weight_decay_amount, model_save_path=model_save_path)

        losses, accuracies_train, accuracies_test = statistics
        # plot the statistics
        if show_statistics:
            plot_train_statistics(x_values=range(len(losses)), y_values=losses, x_label="Iteration", y_label="Loss")
            plot_train_statistics(x_values=range(len(accuracies_train)), y_values=accuracies_train,
                                  x_label="Epoch", y_label="Train accuracy")
            plot_train_statistics(x_values=range(len(accuracies_test)), y_values=accuracies_test,
                                  x_label="Epoch", y_label="Test accuracy", show_plot=True)

    if predict_now:
        model = load_model(model_save_path, amount_of_labels=len(fruit_label_enum),
                           batch_normalization=batch_normalization, dropout=dropout,
                           drop_prob=drop_prob, kernel_size=kernel_size, padding=padding, data_width=data_width,
                           data_height=2, num_channels_layer1=num_channels_layer1,
                           num_channels_layer2=num_channels_layer2)

        confidence, prediction = predict(model=model, data_file=file_to_predict, transform=transform,
                                         fruit_label_enum=fruit_label_enum, data_width=data_width,
                                         confidence_threshold=confidence_threshold)

        print("Prediction: {},\tConfidence: {:.3f}%".format(prediction, confidence*100))
        return confidence, prediction


if __name__ == '__main__':
    main(create_dataset_now=False, num_epochs=20, kernel_size=(2, 5), padding=(1, 5), model_save_path=r"model.pth",
         batch_size=20, train_now=False, predict_now=True, file_to_predict="banana neg.txt")
