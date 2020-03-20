import matplotlib.pyplot as plt
from common import read_data


def plot_on_same_graph(datas, legend_labels, show_plot=False):
    fig, ax = plt.subplots()
    for data in datas:
        if type(data) == dict:
            ax.plot(data["x"], data["y"])
        else:
            ax.plot(data[1], data[0])
    ax.legend(legend_labels)
    ax.set_xlabel('Molecular Weight')
    ax.set_ylabel('Intensity')
    if show_plot:
        plt.show()


def plot_graph(data, legend_label, show_plot=False):
    fig, ax = plt.subplots()
    if type(data) == dict:
        ax.plot(data["x"], data["y"])
    else:
        ax.plot(data[1], data[0])
    ax.legend(legend_label)
    ax.set_xlabel('Molecular Weight')
    ax.set_ylabel('Intensity')
    if show_plot:
        plt.show()

if __name__ == '__main__':
    apple_file = r"apple pos.txt"
    banana_file = r"banana pos.txt"
    apple_data_read = read_data(apple_file)
    banana_data_read = read_data(banana_file)
    plot_graph(apple_data_read, ["Apple"])
    plot_graph(banana_data_read, ["Banana"])
    plot_on_same_graph([apple_data_read, banana_data_read], ["Apple", "Banana"])
    plt.show()