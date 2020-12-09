import copy
import math


class KNN:
    def __init__(self, k, train_data_loader, test_data_loader):
        self.k = k
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def train(self):
        total = 0.0
        correct = 0.0
        true_labels = []
        predictions = []
        for test_spectrum, test_label in copy.deepcopy(self.test_data_loader).load_data():
            distances = self.calculate_distance_of_one_test_data(test_spectrum)
            k_nearest_neighbours = self.get_k_nearest_neighbours(distances)
            predicted_label = self.get_label_from_k_nearest_neighbours(k_nearest_neighbours)
            if predicted_label == test_label[0]:
                correct += 1
            total += 1
            predictions.append(predicted_label)
            true_labels.extend(test_label)
        accuracy = correct / total
        return accuracy, true_labels, predictions

    def calculate_distance_of_one_test_data(self, test_data):
        test_data = test_data.view(-1, )
        # test_y_axis = test_data[1]  # We only care about y data, x is the close enough in all the data
        distances_and_corresponding_label = []
        for train_spectrum, train_labels in copy.deepcopy(self.train_data_loader).load_data():
            train_spectrum = train_spectrum.view(-1, )
            # train_y_axis = train_spectrum[1]
            distance = train_spectrum - test_data
            distance *= distance  # get the square for pithagoras
            distance = distance.sum()
            distances_and_corresponding_label.append((math.sqrt(distance), train_labels[0]))
        return distances_and_corresponding_label

    def get_k_nearest_neighbours(self, distances):
        distances.sort(key=lambda distances_labels: distances_labels[0])
        return distances[:self.k]

    def get_label_from_k_nearest_neighbours(self, k_nearest_labels):
        histogram = {}
        for distance, label in k_nearest_labels:
            histogram[label] = histogram.get(label, 0) + 1

        max_appearance = "foo"
        for label in histogram:
            if histogram[label] > histogram.get(max_appearance, 0):
                max_appearance = label
        return max_appearance
