import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sklearn.metrics as sk_metrics
import pandas as pd
import numpy as np
import string
import seaborn as sn
from scipy.interpolate import make_interp_spline, BSpline

from common import read_data


def plot_raw_data_on_same_graph(datas, legend_labels, show_plot=False):
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


def plot_raw_data(data, legend_label, show_plot=False):
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


def __get_axis_ticks_for_confusion_matrix(fruits, true_labels=None):
    """Get the labels of the axis (the x ticks and y ticks) to be either the fruits,
    or just the default class A, class B, etc."""
    if fruits is None and true_labels is None:
        raise ValueError("Error in getting axis ticks for confusion matrix.\n"
                         "If fruits is None true_labels must not be None")
    if fruits:
        axis_ticks = list(fruits)
    else:  # fruits is None
        axis_ticks = ["class %s" %i for i in list(string.ascii_uppercase)[0:len(np.unique(true_labels))]]

    return axis_ticks


def __insert_totals(df_confusion_matrix):
    """Insert `total` column and row"""
    sum_col = []
    for col in df_confusion_matrix.columns:
        sum_col.append(df_confusion_matrix[col].sum())
    sum_row = []
    for row in df_confusion_matrix.iterrows():
        sum_row.append(row[1].sum())

    df_confusion_matrix["total row"] = sum_row
    sum_col.append(np.sum(sum_row))
    df_confusion_matrix.loc['total column'] = sum_col


def __build_confusion_matrix(true_labels, predictions, fruits=None, insert_totals=True):
    """build a confusion matrix that later can be plotted"""
    # Getting the raw confusion matrix
    # This by itself is enough for plotting, but we want to make ir prettier
    confusion_matrix = sk_metrics.confusion_matrix(y_true=true_labels, y_pred=predictions)

    # Get the labels of the axis (the x ticks and y ticks) to be either the fruits
    # or just the default class A, class B, etc.
    axis_ticks = __get_axis_ticks_for_confusion_matrix(fruits=fruits, true_labels=true_labels)

    # Make it a pandas data frame, it will be easier to plot that way
    df_confusion_matrix = pd.DataFrame(data=confusion_matrix, index=axis_ticks, columns=axis_ticks)

    # insert the `total` row and column
    if insert_totals:
        __insert_totals(df_confusion_matrix)

    return df_confusion_matrix


# def __config_cell_text_and_colors(array_df, line, col, text, facecolors, posi, fontsize, precision, show_null_values):
def __config_cell_text_and_colors(array_df, line, col, text, posi, fontsize, precision, show_null_values):

    """
      config cell text and colors
      and return text elements to add and to dell
    """
    text_add = []
    text_del = []
    cell_val = array_df[line][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line and/or last column
    if (col == (ccl - 1)) or (line == (ccl - 1)):
        # totals and percents
        if cell_val != 0:
            if (col == ccl - 1) and (line == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[line][line]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif line == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0
        format_string = r"%{}%%".format(precision)
        per_ok_s = [format_string % per_ok, '100%'][per_ok == 100]

        # text to delete
        text_del.append(text)

        # text to add
        font_prop = fm.FontProperties(weight='bold', size=fontsize)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % cell_val, per_ok_s, format_string % per_err]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic['color'] = 'g'
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic['color'] = 'r'
        lis_kwa.append(dic)
        lis_pos = [(text._x, text._y - 0.3), (text._x, text._y), (text._x, text._y + 0.3)]
        for i in range(len(lis_txt)):
            new_text = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            text_add.append(new_text)

        # # Set background color for sum cells (last line and last column)
        # carr = [0.27, 0.30, 0.27, 1.0]
        # if(col == ccl - 1) and (line == ccl - 1):
        #     carr = [0.17, 0.20, 0.17, 1.0]
        # facecolors[posi] = carr

    else:
        if per > 0:
            format_string = r"%s{}%{}%%".format("\n", precision)
            txt = format_string % (cell_val, per)
        else:
            if show_null_values:
                txt = '0\n0%'
            else:
                txt = ''

        text.set_text(txt)

        # main diagonal
        if col == line:
            # set color of the textin the diagonal to white
            text.set_color('r')
            # # set background color in the diagonal to blue
            # facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            text.set_color('r')

    return text_add, text_del


def plot_confusion_matrix(true_labels, predictions, fruits=None, annotate=True, cmap="Oranges", precision=".2f",
                          font_size=11, line_width=0.5, colorbar=False, figsize=(9, 9), show_null_values=False,
                          prediction_axis="y", insert_totals=True, title="Confusion matrix", show_plot=False):

    # Get the confusion matrix data frame
    df_confusion_matrix = __build_confusion_matrix(true_labels=true_labels, predictions=predictions, fruits=fruits,
                                                   insert_totals=insert_totals)

    # set the labels of the axis
    if prediction_axis == "x":
        x_label = "Predicted"
        y_label = "Actual"
    else:  # prediction_axis == "y"
        x_label = "Actual"
        y_label = "Predicted"
        df_confusion_matrix = df_confusion_matrix.T

    # Starting to plot
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()  # get current axis

    ax = sn.heatmap(df_confusion_matrix, annot=annotate, annot_kws={"size": font_size}, linewidths=line_width, ax=ax,
                    cbar=colorbar, cmap=cmap, fmt=precision)

    # set tick labels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible = False
        tick.tick2line.set_visible = False
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible = False
        tick.tick2line.set_visible = False

    # Face colors list
    # quadmesh = ax.findobj(Quadmesh)[0]
    # facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_confusion_matrix.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for text in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(text.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        # txt_res = __config_cell_text_and_colors(array_df, lin, col, text, facecolors, posi, font_size, precision,
        #                                       show_null_values)
        txt_res = __config_cell_text_and_colors(array_df, lin, col, text, posi, font_size, precision,
                                                show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        # ax.text.remove(item)
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()  # set layout slim
    if show_plot:
        plt.show()


def plot_classification_report(true_labels, predictions, bar_width=0.25, title="Classification Report",
                               show_plot=False):
    report = sk_metrics.classification_report(true_labels, predictions)
    report_splitted = report.split("\n")

    # get bars values
    x_ticks = []  # actually also the labels
    precision_bars = []
    recall_bars = []
    f_measure_bars = []
    bar_values_now = False  # flag
    for i, section in enumerate(report_splitted):
        section = [value for value in section.strip().split(" ") if value]
        if i == 1:
            bar_values_now = True
            continue
        if i != 1 and report_splitted[i] == "":
            bar_values_now = False
        if bar_values_now:
            x_ticks.append(section[0])
            precision_bars.append(float(section[1]))
            recall_bars.append(float(section[2]))
            f_measure_bars.append(float(section[3]))
        else:
            if "accuracy" in section:
                accuracy = float(section[1])

    # Set position of bar on X axis
    r1 = np.arange(len(precision_bars))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig = plt.figure()
    ax = fig.gca()  # get current axis

    # Make the plot
    ax.bar(r1, precision_bars, color='#7f6d5f', width=bar_width, edgecolor='white', label='precision')
    ax.bar(r2, recall_bars, color='#557f2d', width=bar_width, edgecolor='white', label='recall')
    ax.bar(r3, f_measure_bars, color='#2d7f5e', width=bar_width, edgecolor='white', label='F-measure')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Accuracy: {:.2f}%'.format(accuracy * 100), fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(precision_bars))])
    ax.set_xticklabels(x_ticks)

    # set title
    ax.set_title(title)

    # Create legend & Show graphic
    ax.legend()
    if show_plot:
        plt.show()


def plot_train_statistics(x_values, y_values, x_label, y_label, show_plot=False, interpolate_spline=True):
    fig, ax = plt.subplots()
    if interpolate_spline:
        x_smooth = np.linspace(min(x_values), max(x_values), 300)
        spline = make_interp_spline(x_values, y_values, k=3)  # type: BSpline
        y_values = spline(x_smooth)
        x_values = x_smooth
    ax.plot(x_values, y_values)
    # ax.legend(legend_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    # plot raw data
    apple_file = r"C:\Users\Nadav\Desktop\12_Kai\Apple\Anal\After 5\YY20-173 neg.txt"
    banana_file = r"C:\Users\Nadav\Desktop\12_Kai\Banana\Anal\After 5\YY20-085 neg.txt"
    apple_data_read = read_data(apple_file)
    banana_data_read = read_data(banana_file)
    plot_raw_data(apple_data_read, ["Apple"])
    plot_raw_data(banana_data_read, ["Banana"])
    plot_raw_data_on_same_graph([apple_data_read, banana_data_read], ["Apple", "Banana"])

    # plot confusion matrix
    true_labels = np.array(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
         3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
         5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    predictions = np.array(
        [1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 4, 4, 1, 4, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2,
         4, 4, 5, 1, 2, 3, 3, 5, 1, 2, 3, 3, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 4, 4,
         5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    fruits = ["fruit 1", "fruit 2", "fruit 3", "fruit 4", "fruit 5"]
    plot_confusion_matrix(true_labels=true_labels, predictions=predictions, fruits=fruits, title="Confusion matrix",
                          show_null_values=True)

    plot_classification_report(true_labels=true_labels, predictions=predictions)
    plt.show()
