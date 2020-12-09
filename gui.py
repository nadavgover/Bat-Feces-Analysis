from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
import os
import threading

from main import main

MODEL_PATH = 0
PREDICT_DATA_PATH = 1
ROOT_DIR = 2
TRAIN_SPECTRUM_PATH = 3
TRAIN_LABELS_PATH = 4
TEST_SPECTRUM_PATH = 5
TEST_LABELS_PATH = 6
DEFAULTS = {"model_path": "model.pth", "fruits": "apple, banana, mix", "kernel_size": "(1, 2)", "padding": "(0, 0)",
            "data_width": "2100", "confidence_threshold": "0.7", "root_dir": "YOMIRAN",
            "sample_times": ["after 5", "after 8", "before", "after 5, after 8", "after 5, before",
                             "after 8, before", "all"],
            "sample_locations": ["anal", "oral", "all"], "sample_types": ["pos", "neg", "all"],
            "train_spectrum_path": "train_spectrum",
            "test_spectrum_path": "test_spectrum", "train_labels_path": "train_labels",
            "test_labels_path": "test_labels", "dataset_size": "10000", "train_data_percentage": "0.8",
            "dataset_folder_name": "dataset", "fc1_amount_output_nodes": "100", "fc2_amount_output_nodes": "500",
            "fc3_amount_output_nodes": "100", "num_channels_layer1": "30", "num_channels_layer2": "6",
            "batch_normalization": "True", "drop_prob": 0.2, "epoch_num": 50}


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window"""
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 17
        y = y + cy + self.widget.winfo_rooty() + 17
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create_tooltip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


class FinalProjectGui(Tk):
    def __init__(self):
        super(FinalProjectGui, self).__init__()
        self.title("Bat Feces Analyzer")
        self.minsize(400, 400)
        self.maxsize(400, 400)
        self.wm_iconbitmap('bat.ico')

        # self.kernel_size = DEFAULTS["kernel_size"]
        # self.padding = DEFAULTS["padding"]
        # self.confidence_threshold = DEFAULTS["confidence_threshold"]
        self.tooltip_icon = PhotoImage(file="tooltip-black-16x16.png")

        # Adding tabs
        self.main_tab_parent = ttk.Notebook(self)
        self.predict_tab = ttk.Frame(self.main_tab_parent)
        self.create_dataset_tab = ttk.Frame(self.main_tab_parent)
        self.train_tab = ttk.Frame(self.main_tab_parent)
        self.main_tab_parent.add(self.predict_tab, text='Predict')
        self.main_tab_parent.add(self.create_dataset_tab, text='Create Dataset')
        self.main_tab_parent.add(self.train_tab, text='Train')
        self.main_tab_parent.pack(expand=1, fill="both")

        self.design_predict_tab()
        self.design_create_dataset_tab()
        self.design_train_tab()

    def design_train_tab(self):
        self.tabs_in_train_parent = ttk.Notebook(self.train_tab)
        self.settings_train_tab = ttk.Frame(self.tabs_in_train_parent)
        self.advanced_train_tab = ttk.Frame(self.tabs_in_train_parent)
        self.tabs_in_train_parent.add(self.settings_train_tab, text="Settings")
        self.tabs_in_train_parent.add(self.advanced_train_tab, text="Advanced")
        self.tabs_in_train_parent.pack(expand=1, fill="both")

        self.design_train_settings_tab()
        self.design_train_advanced_tab()

        # train button
        self.train_button = ttk.Button(self.tabs_in_train_parent, text="Train",
                                       command=self.train)

        self.train_button.place(relx=0.5, rely=0.90, anchor=CENTER)

    def design_train_settings_tab(self):
        # train spectrum file
        self.train_spectrum_file_frame_in_train = ttk.Frame(self.settings_train_tab, borderwidth=2, relief=SUNKEN)
        self.train_spectrum_file_frame_in_train.place(relx=0.65, rely=0.08, anchor=CENTER)
        self.train_spectrum_file_in_train_label = ttk.Label(self.settings_train_tab, text="Train spectrum file:")
        self.train_spectrum_file_in_train_label.place(relx=0.2, rely=0.08, anchor=CENTER)
        self.train_spectrum_file_in_train_stringvar = StringVar()
        self.train_spectrum_file_in_train_entry = Entry(self.train_spectrum_file_frame_in_train,
                                                        textvariable=self.train_spectrum_file_in_train_stringvar,
                                                        justify=CENTER, borderwidth=0, highlightthickness=1,
                                                        highlightbackground="grey", background="white")
        self.train_spectrum_file_in_train_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        train_spectrum_file_dialog = partial(self.file_dialog, data_or_model_file_path=TRAIN_SPECTRUM_PATH)
        self.train_spectrum_browse_button_in_train = ttk.Button(self.train_spectrum_file_frame_in_train, text="Browse",
                                                                command=train_spectrum_file_dialog)
        self.train_spectrum_browse_button_in_train.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # train labels file
        self.train_labels_file_frame_in_train = ttk.Frame(self.settings_train_tab, borderwidth=2, relief=SUNKEN)
        self.train_labels_file_frame_in_train.place(relx=0.65, rely=0.19, anchor=CENTER)
        self.train_labels_file_in_train_label = ttk.Label(self.settings_train_tab, text="Train labels file:")
        self.train_labels_file_in_train_label.place(relx=0.175, rely=0.19, anchor=CENTER)
        self.train_labels_file_in_train_stringvar = StringVar()
        self.train_labels_file_in_train_entry = Entry(self.train_labels_file_frame_in_train,
                                                      textvariable=self.train_labels_file_in_train_stringvar,
                                                      justify=CENTER, borderwidth=0, highlightthickness=1,
                                                      highlightbackground="grey", background="white")
        self.train_labels_file_in_train_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        train_labels_file_dialog = partial(self.file_dialog, data_or_model_file_path=TRAIN_LABELS_PATH)
        self.train_labels_browse_button_in_train = ttk.Button(self.train_labels_file_frame_in_train, text="Browse",
                                                              command=train_labels_file_dialog)
        self.train_labels_browse_button_in_train.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # test spectrum file
        self.test_spectrum_file_frame_in_train = ttk.Frame(self.settings_train_tab, borderwidth=2, relief=SUNKEN)
        self.test_spectrum_file_frame_in_train.place(relx=0.65, rely=0.30, anchor=CENTER)
        self.test_spectrum_file_in_train_label = ttk.Label(self.settings_train_tab, text="Test spectrum file:")
        self.test_spectrum_file_in_train_label.place(relx=0.195, rely=0.30, anchor=CENTER)
        self.test_spectrum_file_in_train_stringvar = StringVar()
        self.test_spectrum_file_in_train_entry = Entry(self.test_spectrum_file_frame_in_train,
                                                       textvariable=self.test_spectrum_file_in_train_stringvar,
                                                       justify=CENTER, borderwidth=0, highlightthickness=1,
                                                       highlightbackground="grey", background="white")
        self.test_spectrum_file_in_train_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        test_spectrum_file_dialog = partial(self.file_dialog, data_or_model_file_path=TEST_SPECTRUM_PATH)
        self.test_spectrum_browse_button_in_train = ttk.Button(self.test_spectrum_file_frame_in_train, text="Browse",
                                                               command=test_spectrum_file_dialog)
        self.test_spectrum_browse_button_in_train.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # test labels file
        self.test_labels_file_frame_in_train = ttk.Frame(self.settings_train_tab, borderwidth=2, relief=SUNKEN)
        self.test_labels_file_frame_in_train.place(relx=0.65, rely=0.41, anchor=CENTER)
        self.test_labels_file_in_train_label = ttk.Label(self.settings_train_tab, text="Test labels file:")
        self.test_labels_file_in_train_label.place(relx=0.17, rely=0.41, anchor=CENTER)
        self.test_labels_file_in_train_stringvar = StringVar()
        self.test_labels_file_in_train_entry = Entry(self.test_labels_file_frame_in_train,
                                                     textvariable=self.test_labels_file_in_train_stringvar,
                                                     justify=CENTER, borderwidth=0, highlightthickness=1,
                                                     highlightbackground="grey", background="white")
        self.test_labels_file_in_train_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        test_labels_file_dialog = partial(self.file_dialog, data_or_model_file_path=TEST_LABELS_PATH)
        self.test_labels_browse_button_in_train = ttk.Button(self.test_labels_file_frame_in_train, text="Browse",
                                                             command=test_labels_file_dialog)
        self.test_labels_browse_button_in_train.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # epochs number
        self.epoch_num_in_train_label = ttk.Label(self.settings_train_tab, text="Number of epochs: ")
        self.epoch_num_in_train_label.place(relx=0.21, rely=0.52, anchor=CENTER)
        self.epoch_num_in_train_stringvar = StringVar()
        self.epoch_num_in_train_stringvar.set(DEFAULTS["epoch_num"])
        self.epoch_num_in_train_entry = Entry(self.settings_train_tab, justify=CENTER,
                                              borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                              textvariable=self.epoch_num_in_train_stringvar, bg="white")
        self.epoch_num_in_train_entry.place(relx=0.545, rely=0.52, anchor=CENTER)

        # saved model path
        self.saved_model_path_in_train_label = ttk.Label(self.settings_train_tab, text="Saved model path: ")
        self.saved_model_path_in_train_label.place(relx=0.205, rely=0.63, anchor=CENTER)
        self.saved_model_path_in_train_stringvar = StringVar()
        self.saved_model_path_in_train_stringvar.set(DEFAULTS["model_path"])
        self.saved_model_path_in_train_entry = Entry(self.settings_train_tab, justify=CENTER,
                                                     borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                     textvariable=self.saved_model_path_in_train_stringvar, bg="white")
        self.saved_model_path_in_train_entry.place(relx=0.545, rely=0.63, anchor=CENTER)
        # saved model path tooltip
        self.saved_model_path_in_train_tooltip = ttk.Label(self.settings_train_tab)
        self.saved_model_path_in_train_tooltip.image = self.tooltip_icon
        self.saved_model_path_in_train_tooltip["image"] = self.saved_model_path_in_train_tooltip.image
        self.saved_model_path_in_train_tooltip.place(relx=0.03, rely=0.63, anchor=CENTER)
        create_tooltip(widget=self.saved_model_path_in_train_tooltip, text="This file is where the trained model will "
                                                                           "be saved.\nThe extension of the file"
                                                                           " will be .pth")

        # fruits list
        self.fruits_in_train_label = ttk.Label(self.settings_train_tab,
                                               text="Enter fruits:")
        self.fruits_in_train_label.place(relx=0.155, rely=0.74, anchor=CENTER)
        validate_fruit_list_cmd = (self.register(self.validate_fruit_list))  # validate command
        # self.fruit_list_stringvar = StringVar()
        self.fruit_list_in_train_entry = Entry(self.settings_train_tab, justify=CENTER,
                                               borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                               textvariable=self.fruit_list_stringvar, bg="white",
                                               validate='all', validatecommand=(validate_fruit_list_cmd, '%P'))
        self.fruit_list_in_train_entry.place(relx=0.545, rely=0.74, anchor=CENTER)
        self.validate_fruit_list(text="-1")  # hack to fill the default value


    def design_train_advanced_tab(self):
        # batch normalization
        self.check_button_batch_norm_in_train_intvar = IntVar()
        self.check_button_batch_norm_in_train_intvar.set(1)
        self.check_button_batch_norm_in_train = ttk.Checkbutton(self.advanced_train_tab,
                                                                  text="Batch normalization",
                                                                  variable=self.check_button_batch_norm_in_train_intvar)
        self.check_button_batch_norm_in_train.place(relx=0.237, rely=0.07, anchor=CENTER)
        # batch normalization tooltip
        self.check_button_batch_norm_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.check_button_batch_norm_in_train_tooltip.image = self.tooltip_icon
        self.check_button_batch_norm_in_train_tooltip["image"] = self.check_button_batch_norm_in_train_tooltip.image
        self.check_button_batch_norm_in_train_tooltip.place(relx=0.04, rely=0.07, anchor=CENTER)
        create_tooltip(widget=self.check_button_batch_norm_in_train_tooltip, text="Check this to train the model "
                                                                                  "with batch normalization")

        # kernel size
        self.kernel_size_in_train_label = ttk.Label(self.advanced_train_tab, text="Kernel size: ")
        self.kernel_size_in_train_label.place(relx=0.156, rely=0.15, anchor=CENTER)
        self.kernel_size_in_train_stringvar = StringVar()
        self.kernel_size_in_train_stringvar.set(DEFAULTS["kernel_size"])
        self.kernel_size_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                  borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                  textvariable=self.kernel_size_in_train_stringvar, bg="white")
        self.kernel_size_in_train_entry.place(relx=0.685, rely=0.15, anchor=CENTER)
        # kernel size tooltip
        self.kernel_size_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.kernel_size_in_train_tooltip.image = self.tooltip_icon
        self.kernel_size_in_train_tooltip["image"] = self.kernel_size_in_train_tooltip.image
        self.kernel_size_in_train_tooltip.place(relx=0.04, rely=0.15, anchor=CENTER)
        create_tooltip(widget=self.kernel_size_in_train_tooltip, text="Kernel size to use while training the model.\n"
                                                                      "Expecting two natural numbers, surrounded with"
                                                                      " parenthesis, separated with a comma.")

        # padding
        self.padding_in_train_label = ttk.Label(self.advanced_train_tab, text="Padding: ")
        self.padding_in_train_label.place(relx=0.144, rely=0.23, anchor=CENTER)
        self.padding_in_train_stringvar = StringVar()
        self.padding_in_train_stringvar.set(DEFAULTS["padding"])
        self.padding_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                            borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                            textvariable=self.padding_in_train_stringvar, bg="white")
        self.padding_in_train_entry.place(relx=0.685, rely=0.23, anchor=CENTER)
        # padding tooltip
        self.padding_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.padding_in_train_tooltip.image = self.tooltip_icon
        self.padding_in_train_tooltip["image"] = self.padding_in_train_tooltip.image
        self.padding_in_train_tooltip.place(relx=0.04, rely=0.23, anchor=CENTER)
        create_tooltip(widget=self.padding_in_train_tooltip, text="Padding to use while training the model.\n"
                                                                  "Expecting two natural numbers, surrounded with "
                                                                  "parenthesis, separated with a comma.")

        # num channels layer 1
        self.num_channels_layer1_in_train_label = ttk.Label(self.advanced_train_tab, text="Number of channels in "
                                                                                          "layer 1: ")
        self.num_channels_layer1_in_train_label.place(relx=0.287, rely=0.31, anchor=CENTER)
        self.num_channels_layer1_in_train_stringvar = StringVar()
        self.num_channels_layer1_in_train_stringvar.set(DEFAULTS["num_channels_layer1"])
        self.num_channels_layer1_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                          borderwidth=0, highlightthickness=1,
                                                          highlightbackground="grey",
                                                          textvariable=self.num_channels_layer1_in_train_stringvar,
                                                          bg="white")
        self.num_channels_layer1_in_train_entry.place(relx=0.685, rely=0.31, anchor=CENTER)
        # num channels layer 1 tooltip
        self.num_channels_layer1_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.num_channels_layer1_in_train_tooltip.image = self.tooltip_icon
        self.num_channels_layer1_in_train_tooltip["image"] = self.num_channels_layer1_in_train_tooltip.image
        self.num_channels_layer1_in_train_tooltip.place(relx=0.04, rely=0.31, anchor=CENTER)
        create_tooltip(widget=self.num_channels_layer1_in_train_tooltip, text="Number of channels in layer 1 to use "
                                                                              "while training the model.\n"
                                                                              "Layer 1 is a convolutional layer.\n"
                                                                              "Expecting natural number.")

        # num channels layer 2
        self.num_channels_layer2_in_train_label = ttk.Label(self.advanced_train_tab, text="Number of channels in "
                                                                                          "layer 2: ")
        self.num_channels_layer2_in_train_label.place(relx=0.287, rely=0.39, anchor=CENTER)
        self.num_channels_layer2_in_train_stringvar = StringVar()
        self.num_channels_layer2_in_train_stringvar.set(DEFAULTS["num_channels_layer2"])
        self.num_channels_layer2_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                        borderwidth=0, highlightthickness=1,
                                                        highlightbackground="grey",
                                                        textvariable=self.num_channels_layer2_in_train_stringvar,
                                                        bg="white")
        self.num_channels_layer2_in_train_entry.place(relx=0.685, rely=0.39, anchor=CENTER)
        # num channels layer 2 tooltip
        self.num_channels_layer2_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.num_channels_layer2_in_train_tooltip.image = self.tooltip_icon
        self.num_channels_layer2_in_train_tooltip["image"] = self.num_channels_layer2_in_train_tooltip.image
        self.num_channels_layer2_in_train_tooltip.place(relx=0.04, rely=0.39, anchor=CENTER)
        create_tooltip(widget=self.num_channels_layer2_in_train_tooltip, text="Number of channels in layer 2 to use "
                                                                              "while training the model.\n"
                                                                              "Layer 2 is a convolutional layer.\n"
                                                                              "Expecting a natural number.")

        # amount output nodes fc1
        self.num_output_nodes_fc1_in_train_label = ttk.Label(self.advanced_train_tab, text="Number of output nodes "
                                                                                           "layer 3: ")
        self.num_output_nodes_fc1_in_train_label.place(relx=0.298, rely=0.47, anchor=CENTER)
        self.num_output_nodes_fc1_in_train_stringvar = StringVar()
        self.num_output_nodes_fc1_in_train_stringvar.set(DEFAULTS["fc1_amount_output_nodes"])
        self.num_output_nodes_fc1_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                         borderwidth=0, highlightthickness=1,
                                                         highlightbackground="grey",
                                                         textvariable=self.num_output_nodes_fc1_in_train_stringvar,
                                                         bg="white")
        self.num_output_nodes_fc1_in_train_entry.place(relx=0.685, rely=0.47, anchor=CENTER)
        # amount output nodes fc1 tooltip
        self.num_output_nodes_fc1_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.num_output_nodes_fc1_in_train_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc1_in_train_tooltip["image"] = self.num_output_nodes_fc1_in_train_tooltip.image
        self.num_output_nodes_fc1_in_train_tooltip.place(relx=0.04, rely=0.47, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc1_in_train_tooltip, text="Number of output nodes in layer 3 "
                                                                               "to use while training the model.\n"
                                                                               "Layer 3 is a fully connected layer.\n"
                                                                               "Expecting a natural number.")

        # amount output nodes fc2
        self.num_output_nodes_fc2_in_train_label = ttk.Label(self.advanced_train_tab, text="Number of output nodes "
                                                                                           "layer 4: ")
        self.num_output_nodes_fc2_in_train_label.place(relx=0.298, rely=0.55, anchor=CENTER)
        self.num_output_nodes_fc2_in_train_stringvar = StringVar()
        self.num_output_nodes_fc2_in_train_stringvar.set(DEFAULTS["fc2_amount_output_nodes"])
        self.num_output_nodes_fc2_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                         borderwidth=0, highlightthickness=1,
                                                         highlightbackground="grey",
                                                         textvariable=self.num_output_nodes_fc2_in_train_stringvar,
                                                         bg="white")
        self.num_output_nodes_fc2_in_train_entry.place(relx=0.685, rely=0.55, anchor=CENTER)
        # amount output nodes fc2 tooltip
        self.num_output_nodes_fc2_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.num_output_nodes_fc2_in_train_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc2_in_train_tooltip["image"] = self.num_output_nodes_fc2_in_train_tooltip.image
        self.num_output_nodes_fc2_in_train_tooltip.place(relx=0.04, rely=0.55, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc2_in_train_tooltip, text="Number of output nodes in layer 4 "
                                                                               "to use while training the model.\n"
                                                                               "Layer 4 is a fully connected layer.\n"
                                                                               "Expecting a natural number.")

        # amount output nodes fc3
        self.num_output_nodes_fc3_in_train_label = ttk.Label(self.advanced_train_tab, text="Number of output nodes "
                                                                                           "layer 5: ")
        self.num_output_nodes_fc3_in_train_label.place(relx=0.298, rely=0.63, anchor=CENTER)
        self.num_output_nodes_fc3_in_train_stringvar = StringVar()
        self.num_output_nodes_fc3_in_train_stringvar.set(DEFAULTS["fc3_amount_output_nodes"])
        self.num_output_nodes_fc3_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                                         borderwidth=0, highlightthickness=1,
                                                         highlightbackground="grey",
                                                         textvariable=self.num_output_nodes_fc3_in_train_stringvar,
                                                         bg="white")
        self.num_output_nodes_fc3_in_train_entry.place(relx=0.685, rely=0.63, anchor=CENTER)
        # amount output nodes fc3 tooltip
        self.num_output_nodes_fc3_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.num_output_nodes_fc3_in_train_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc3_in_train_tooltip["image"] = self.num_output_nodes_fc3_in_train_tooltip.image
        self.num_output_nodes_fc3_in_train_tooltip.place(relx=0.04, rely=0.63, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc3_in_train_tooltip, text="Number of output nodes in layer 5 "
                                                                               "to use while training the model.\n"
                                                                               "Layer 5 is a fully connected layer.\n"
                                                                               "Expecting a natural number.")

        # dropout
        self.check_button_dropout_in_train_intvar = IntVar()
        self.check_button_dropout_in_train_intvar.set(1)
        self.check_button_dropout_in_train = ttk.Checkbutton(self.advanced_train_tab,
                                                             text="Dropout",
                                                             variable=self.check_button_dropout_in_train_intvar)
        self.check_button_dropout_in_train.place(relx=0.157, rely=0.71, anchor=CENTER)
        # dropout tooltip
        self.check_button_dropout_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.check_button_dropout_in_train_tooltip.image = self.tooltip_icon
        self.check_button_dropout_in_train_tooltip["image"] = self.check_button_dropout_in_train_tooltip.image
        self.check_button_dropout_in_train_tooltip.place(relx=0.04, rely=0.71, anchor=CENTER)
        create_tooltip(widget=self.check_button_dropout_in_train_tooltip, text="Check this to train the model "
                                                                               "with dropout")

        # drop probability
        self.drop_prob_in_train_label = ttk.Label(self.advanced_train_tab, text="Dropout probability")
        self.drop_prob_in_train_label.place(relx=0.212, rely=0.79, anchor=CENTER)
        self.drop_prob_in_train_stringvar = StringVar()
        self.drop_prob_in_train_stringvar.set(DEFAULTS["drop_prob"])
        self.drop_prob_in_train_entry = Entry(self.advanced_train_tab, justify=CENTER,
                                              borderwidth=0, highlightthickness=1,
                                              highlightbackground="grey",
                                              textvariable=self.drop_prob_in_train_stringvar,
                                              bg="white")
        self.drop_prob_in_train_entry.place(relx=0.685, rely=0.79, anchor=CENTER)
        # drop prob tooltip
        self.drop_prob_in_train_tooltip = ttk.Label(self.advanced_train_tab)
        self.drop_prob_in_train_tooltip.image = self.tooltip_icon
        self.drop_prob_in_train_tooltip["image"] = self.drop_prob_in_train_tooltip.image
        self.drop_prob_in_train_tooltip.place(relx=0.04, rely=0.79, anchor=CENTER)
        create_tooltip(widget=self.drop_prob_in_train_tooltip, text="Drop probability to use while training\n"
                                                                    "Relevant only if dropout is set\n"
                                                                    "Expecting a number between 0-1")

    def train(self):
        if not self.is_valid_train_inputs():
            return

        train_spectrum_path = self.train_spectrum_file_in_train_stringvar.get()
        train_labels_path = self.train_labels_file_in_train_stringvar.get()
        test_spectrum_path = self.test_spectrum_file_in_train_stringvar.get()
        test_labels_path = self.test_labels_file_in_train_stringvar.get()

        fruits = self.fruit_list_in_train_entry.get().split(sep=",")
        fruits = [fruit.strip() for fruit in fruits]

        kernel_size_left, kernel_size_right = self.kernel_size_in_train_stringvar.get().split(",")
        kernel_size = (int(kernel_size_left.strip(" (")), int(kernel_size_right.strip(" )")))

        padding_left, padding_right = self.padding_in_train_stringvar.get().split(",")
        padding = (int(padding_left.strip(" (")), int(padding_right.strip(" )")))

        num_channels_layer1 = int(self.num_channels_layer1_in_train_stringvar.get())
        num_channels_layer2 = int(self.num_channels_layer2_in_train_stringvar.get())

        num_output_nodes_fc1 = int(self.num_output_nodes_fc1_in_train_stringvar.get())
        num_output_nodes_fc2 = int(self.num_output_nodes_fc2_in_train_stringvar.get())
        num_output_nodes_fc3 = int(self.num_output_nodes_fc3_in_train_stringvar.get())

        batch_normalization = bool(self.check_button_batch_norm_in_train_intvar.get())

        dropout = bool(self.check_button_dropout_in_train_intvar.get())
        drop_prob = float(self.drop_prob_in_train_stringvar.get())

        epoch_num = int(self.epoch_num_in_train_stringvar.get())

        saved_model_path = self.saved_model_path_in_train_stringvar.get()
        if not saved_model_path.endswith(".pth"):
            saved_model_path = "".join([saved_model_path, ".pth"])

        # progress bar
        self.train_button["state"] = "disable"
        self.train_progress_bar_intvar = IntVar()
        self.train_progress_bar_intvar.set(0)
        self.train_progress_bar = ttk.Progressbar(self.train_tab, orient=HORIZONTAL,
                                                  length=250, mode="indeterminate",
                                                  variable=self.train_progress_bar_intvar,
                                                  maximum=100)
        self.train_progress_bar.place(relx=0.5, rely=0.775, anchor=CENTER)

        train_thread_function = partial(main, train_now=True, fruits=fruits,
                                        train_spectrum_path=train_spectrum_path,
                                        train_labels_path=train_labels_path,
                                        test_spectrum_path=test_spectrum_path,
                                        test_labels_path=test_labels_path,
                                        batch_normalization=batch_normalization,
                                        dropout=dropout, drop_prob=drop_prob, kernel_size=kernel_size,
                                        padding=padding,
                                        num_channels_layer1=num_channels_layer1,
                                        num_channels_layer2=num_channels_layer2,
                                        fc1_amount_output_nodes=num_output_nodes_fc1,
                                        fc2_amount_output_nodes=num_output_nodes_fc2,
                                        fc3_amount_output_node=num_output_nodes_fc3,
                                        model_save_path=saved_model_path,
                                        num_epochs=epoch_num,
                                        batch_size=1, learning_rate=0.01,
                                        knn=False,
                                        show_statistics=False)
        self.train_thread = threading.Thread(target=train_thread_function)
        self.train_thread.start()
        self.after(50, self.update_labels_of_progress_bar, self.train_thread,
                   self.train_progress_bar,
                   self.train_button, self.train_progress_bar_intvar, "Train",
                   "Training finished successfully")

    def design_create_dataset_tab(self):
        self.tabs_in_create_dataset_parent = ttk.Notebook(self.create_dataset_tab)
        self.settings_create_dataset_tab = ttk.Frame(self.tabs_in_create_dataset_parent)
        self.advanced_create_dataset_tab = ttk.Frame(self.tabs_in_create_dataset_parent)
        self.tabs_in_create_dataset_parent.add(self.settings_create_dataset_tab, text="Settings")
        self.tabs_in_create_dataset_parent.add(self.advanced_create_dataset_tab, text="Advanced")
        self.tabs_in_create_dataset_parent.pack(expand=1, fill="both")

        self.design_create_dataset_settings_tab()
        self.design_create_dataset_advanced_tab()

    def design_create_dataset_settings_tab(self):
        # root dir
        self.rootdir_label_frame_in_create_dataset = ttk.LabelFrame(self.settings_create_dataset_tab,
                                                                    text="Folder containing all data files")
        self.rootdir_label_frame_in_create_dataset.place(relx=0.5, rely=0.13, anchor=CENTER)
        self.rootdir_in_create_dataset_label = ttk.Label(self.rootdir_label_frame_in_create_dataset, text="")
        self.rootdir_in_create_dataset_label.grid(column=1, row=2)
        if os.path.exists(DEFAULTS["root_dir"]):
            self.root_dir = os.path.join(os.getcwd(), DEFAULTS["root_dir"])
            self.rootdir_in_create_dataset_label.configure(text=self.root_dir)
        root_dir_dialog = partial(self.file_dialog, data_or_model_file_path=ROOT_DIR)
        self.rootdir_browse_button_in_create_dataset = ttk.Button(self.rootdir_label_frame_in_create_dataset,
                                                                  text="Browse", command=root_dir_dialog)
        self.rootdir_browse_button_in_create_dataset.grid(column=1, row=1)

        # fruits list
        self.fruits_in_create_dataset_label_frame = ttk.LabelFrame(self.settings_create_dataset_tab, text="Fruits")
        self.fruits_in_create_dataset_label_frame.place(relx=0.5, rely=0.35, anchor=CENTER)
        self.fruits_in_create_dataset_label = ttk.Label(self.fruits_in_create_dataset_label_frame,
                                                        text="Please enter fruits (comma separated)")
        self.fruits_in_create_dataset_label.grid(column=1, row=1)
        validate_fruit_list_cmd = (self.register(self.validate_fruit_list))  # validate command
        # self.fruit_list_stringvar = StringVar()  # this is already defined
        self.fruit_list_in_create_dataset_entry = Entry(self.fruits_in_create_dataset_label_frame, justify=CENTER,
                                                        borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                        textvariable=self.fruit_list_stringvar, bg="white",
                                                        validate='all', validatecommand=(validate_fruit_list_cmd, '%P'))
        self.fruit_list_in_create_dataset_entry.grid(column=1, row=2)
        self.validate_fruit_list(text="-1")  # hack to fill the default value

        # sample time
        self.sample_time_in_create_dataset_label = ttk.Label(self.settings_create_dataset_tab, text="Sample time")
        self.sample_time_in_create_dataset_label.place(relx=0.28, rely=0.49, anchor=CENTER)
        self.sample_time_combo_box = ttk.Combobox(self.settings_create_dataset_tab, state="readonly",
                                                  values=DEFAULTS["sample_times"])
        self.sample_time_combo_box.set(DEFAULTS["sample_times"][0])
        self.sample_time_combo_box.place(relx=0.28, rely=0.55, anchor=CENTER)

        # sample location
        self.sample_location_in_create_dataset_label = ttk.Label(self.settings_create_dataset_tab,
                                                                 text="Sample location")
        self.sample_location_in_create_dataset_label.place(relx=0.72, rely=0.49, anchor=CENTER)
        self.sample_location_combo_box = ttk.Combobox(self.settings_create_dataset_tab, state="readonly",
                                                      values=DEFAULTS["sample_locations"])
        self.sample_location_combo_box.set(DEFAULTS["sample_locations"][0])
        self.sample_location_combo_box.place(relx=0.72, rely=0.55, anchor=CENTER)

        # sample type
        self.sample_type_in_create_dataset_label = ttk.Label(self.settings_create_dataset_tab,
                                                                 text="Sample type")
        self.sample_type_in_create_dataset_label.place(relx=0.5, rely=0.64, anchor=CENTER)
        self.sample_type_combo_box = ttk.Combobox(self.settings_create_dataset_tab, state="readonly",
                                                      values=DEFAULTS["sample_types"])
        self.sample_type_combo_box.set(DEFAULTS["sample_types"][0])
        self.sample_type_combo_box.place(relx=0.5, rely=0.7, anchor=CENTER)

        # create dataset button
        self.create_dataset_button = ttk.Button(self.tabs_in_create_dataset_parent, text="Create Dataset",
                                                command=self.create_dataset)

        self.create_dataset_button.place(relx=0.5, rely=0.90, anchor=CENTER)

    def design_create_dataset_advanced_tab(self):
        # create dataset folder
        self.check_button_create_folder_for_dataset_intvar = IntVar()
        self.check_button_create_folder_for_dataset_intvar.set(1)
        self.check_button_create_folder_for_dataset = ttk.Checkbutton(self.advanced_create_dataset_tab,
                                                                      text="Create a folder for the dataset",
                                                                      variable=self.check_button_create_folder_for_dataset_intvar)
        self.check_button_create_folder_for_dataset.place(relx=0.3, rely=0.07, anchor=CENTER)
        # create dataset folder tooltip
        self.create_folder_for_dataset_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.create_folder_for_dataset_tooltip.image = self.tooltip_icon
        self.create_folder_for_dataset_tooltip["image"] = self.create_folder_for_dataset_tooltip.image
        self.create_folder_for_dataset_tooltip.place(relx=0.04, rely=0.07, anchor=CENTER)
        create_tooltip(widget=self.create_folder_for_dataset_tooltip, text="Create a folder named '{}' in {}\n"
                                                                           "This folder will store all the created "
                                                                           "dataset files.".format(DEFAULTS["dataset_folder_name"],
                                                                                                   os.getcwd()))

        # train spectrum path
        self.train_spectrum_path_label = ttk.Label(self.advanced_create_dataset_tab, text="Train spectrum file path: ")
        self.train_spectrum_path_label.place(relx=0.245, rely=0.15, anchor=CENTER)
        self.train_spectrum_path_stringvar = StringVar()
        self.train_spectrum_path_stringvar.set(DEFAULTS["train_spectrum_path"])
        self.train_spectrum_path_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                               borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                               textvariable=self.train_spectrum_path_stringvar, bg="white")
        self.train_spectrum_path_entry.place(relx=0.58, rely=0.15, anchor=CENTER)
        # train spectrum path tooltip
        self.train_spectrum_path_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.train_spectrum_path_tooltip.image = self.tooltip_icon
        self.train_spectrum_path_tooltip["image"] = self.train_spectrum_path_tooltip.image
        self.train_spectrum_path_tooltip.place(relx=0.04, rely=0.15, anchor=CENTER)
        create_tooltip(widget=self.train_spectrum_path_tooltip, text="This file is where the created train spectrum "
                                                                     "dataset will be saved.\nThe extension of the file"
                                                                     " will be .npy, it is a binary file.\nThis file, "
                                                                     "together with the train labels file, make up the"
                                                                     " training dataset.")

        # train labels path
        self.train_labels_path_label = ttk.Label(self.advanced_create_dataset_tab, text="Train labels file path: ")
        self.train_labels_path_label.place(relx=0.22, rely=0.23, anchor=CENTER)
        self.train_labels_path_stringvar = StringVar()
        self.train_labels_path_stringvar.set(DEFAULTS["train_labels_path"])
        self.train_labels_path_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                             borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                             textvariable=self.train_labels_path_stringvar, bg="white")
        self.train_labels_path_entry.place(relx=0.58, rely=0.23, anchor=CENTER)
        # train labels path tooltip
        self.train_labels_path_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.train_labels_path_tooltip.image = self.tooltip_icon
        self.train_labels_path_tooltip["image"] = self.train_labels_path_tooltip.image
        self.train_labels_path_tooltip.place(relx=0.04, rely=0.23, anchor=CENTER)
        create_tooltip(widget=self.train_labels_path_tooltip, text="This file is where the created train labels "
                                                                   "dataset will be saved.\nThe extension of the file"
                                                                   " will be .npy, it is a binary file.\nThis file, "
                                                                   "together with the train spectrum file, make up the"
                                                                   " training dataset.")

        # test spectrum path
        self.test_spectrum_path_label = ttk.Label(self.advanced_create_dataset_tab, text="Test spectrum file path: ")
        self.test_spectrum_path_label.place(relx=0.24, rely=0.31, anchor=CENTER)
        self.test_spectrum_path_stringvar = StringVar()
        self.test_spectrum_path_stringvar.set(DEFAULTS["test_spectrum_path"])
        self.test_spectrum_path_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                              borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                              textvariable=self.test_spectrum_path_stringvar, bg="white")
        self.test_spectrum_path_entry.place(relx=0.58, rely=0.31, anchor=CENTER)
        # test spectrum path tooltip
        self.test_spectrum_path_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.test_spectrum_path_tooltip.image = self.tooltip_icon
        self.test_spectrum_path_tooltip["image"] = self.test_spectrum_path_tooltip.image
        self.test_spectrum_path_tooltip.place(relx=0.04, rely=0.31, anchor=CENTER)
        create_tooltip(widget=self.test_spectrum_path_tooltip, text="This file is where the created test spectrum "
                                                                    "dataset will be saved.\nThe extension of the file"
                                                                    " will be .npy, it is a binary file.\nThis file, "
                                                                    "together with the test labels file, make up the"
                                                                    " test dataset.")

        # test labels path
        self.test_labels_path_label = ttk.Label(self.advanced_create_dataset_tab, text="Test labels file path: ")
        self.test_labels_path_label.place(relx=0.215, rely=0.39, anchor=CENTER)
        self.test_labels_path_stringvar = StringVar()
        self.test_labels_path_stringvar.set(DEFAULTS["test_labels_path"])
        self.test_labels_path_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                            borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                            textvariable=self.test_labels_path_stringvar, bg="white")
        self.test_labels_path_entry.place(relx=0.58, rely=0.39, anchor=CENTER)
        # train labels path tooltip
        self.test_labels_path_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.test_labels_path_tooltip.image = self.tooltip_icon
        self.test_labels_path_tooltip["image"] = self.test_labels_path_tooltip.image
        self.test_labels_path_tooltip.place(relx=0.04, rely=0.39, anchor=CENTER)
        create_tooltip(widget=self.test_labels_path_tooltip, text="This file is where the created test labels "
                                                                  "dataset will be saved.\nThe extension of the file"
                                                                  " will be .npy, it is a binary file.\nThis file, "
                                                                  "together with the test spectrum file, make up the"
                                                                  " test dataset.")

        # train data percentage
        self.train_data_percentage_label = ttk.Label(self.advanced_create_dataset_tab, text="train data percentage : ")
        self.train_data_percentage_label.place(relx=0.24, rely=0.47, anchor=CENTER)
        self.train_data_percentage_stringvar = StringVar()
        validate_train_data_percentage_cmd = (self.register(self.validate_train_data_percentage))  # validate command
        self.train_data_percentage_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                                 borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                 textvariable=self.train_data_percentage_stringvar, bg="white",
                                                 validate='all',
                                                 validatecommand=(validate_train_data_percentage_cmd, '%P'))
        self.train_data_percentage_entry.place(relx=0.58, rely=0.47, anchor=CENTER)
        self.validate_train_data_percentage(text="-1")  # hack to get the default value
        # train data percentage  tooltip
        self.train_data_percentage_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.train_data_percentage_tooltip.image = self.tooltip_icon
        self.train_data_percentage_tooltip["image"] = self.train_data_percentage_tooltip.image
        self.train_data_percentage_tooltip.place(relx=0.04, rely=0.47, anchor=CENTER)
        create_tooltip(widget=self.train_data_percentage_tooltip, text="Determines how much out of the dataset will be "
                                                                       "the training set in percent %\n"
                                                                       "The rest of the data will be part of the "
                                                                       "test set.\n"
                                                                       "The size of the dataset times the train data "
                                                                       "percentage must be divisible by a thousand.")

        # # size of dataset
        # self.size_of_dataset_label = ttk.Label(self.advanced_create_dataset_tab, text="Size of dataset: ")
        # self.size_of_dataset_label.place(relx=0.185, rely=0.55, anchor=CENTER)
        # self.size_of_dataset_stringvar = StringVar()
        # validate_size_of_dataset_cmd = (self.register(self.validate_size_of_dataset))  # validate command
        # self.size_of_dataset_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
        #                                    borderwidth=0, highlightthickness=1, highlightbackground="grey",
        #                                    textvariable=self.size_of_dataset_stringvar, bg="white",
        #                                    validate='all',
        #                                    validatecommand=(validate_size_of_dataset_cmd, '%P'))
        # self.size_of_dataset_entry.place(relx=0.58, rely=0.55, anchor=CENTER)
        # self.validate_size_of_dataset(text="-1")
        # # size of dataset tooltip
        # self.size_of_dataset_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        # self.size_of_dataset_tooltip.image = self.tooltip_icon
        # self.size_of_dataset_tooltip["image"] = self.size_of_dataset_tooltip.image
        # self.size_of_dataset_tooltip.place(relx=0.04, rely=0.55, anchor=CENTER)
        # create_tooltip(widget=self.size_of_dataset_tooltip, text="Dataset size.\nMust be divisible by a thousand.")

    def create_dataset(self):
        if not self.is_valid_create_dataset_inputs():
            return

        fruits = self.fruit_list_in_predict_entry.get().split(sep=",")
        fruits = [fruit.strip() for fruit in fruits]

        sample_time = self.sample_time_combo_box.get().split(",")
        sample_time = [_sample_time.strip() for _sample_time in sample_time]
        if sample_time[0].lower() == "all":
            sample_time = "all"

        sample_location = self.sample_location_combo_box.get().split(",")
        sample_location = [_sample_location.strip() for _sample_location in sample_location]
        if sample_location[0].lower() == "all":
            sample_location = "all"

        sample_type = self.sample_type_combo_box.get()

        create_dataset_folder = self.check_button_create_folder_for_dataset_intvar.get()
        if create_dataset_folder:
            try:
                os.mkdir(path=os.path.join(os.getcwd(), DEFAULTS["dataset_folder_name"]))
            except OSError:
                pass  # folder already exists

        train_spectrum_path = self.train_spectrum_path_entry.get()
        train_spectrum_path = "".join([train_spectrum_path, ".npy"])
        if create_dataset_folder:
            train_spectrum_path = os.path.join(os.getcwd(), DEFAULTS["dataset_folder_name"], train_spectrum_path)
        else:
            train_spectrum_path = os.path.join(os.getcwd(), train_spectrum_path)

        train_labels_path = self.train_labels_path_entry.get()
        train_labels_path = "".join([train_labels_path, ".npy"])
        if create_dataset_folder:
            train_labels_path = os.path.join(os.getcwd(), DEFAULTS["dataset_folder_name"], train_labels_path)
        else:
            train_labels_path = os.path.join(os.getcwd(), train_labels_path)

        test_spectrum_path = self.test_spectrum_path_entry.get()
        test_spectrum_path = "".join([test_spectrum_path, ".npy"])
        if create_dataset_folder:
            test_spectrum_path = os.path.join(os.getcwd(), DEFAULTS["dataset_folder_name"], test_spectrum_path)
        else:
            test_spectrum_path = os.path.join(os.getcwd(), test_spectrum_path)

        test_labels_path = self.test_labels_path_entry.get()
        test_labels_path = "".join([test_labels_path, ".npy"])
        if create_dataset_folder:
            test_labels_path = os.path.join(os.getcwd(), DEFAULTS["dataset_folder_name"], test_labels_path)
        else:
            test_labels_path = os.path.join(os.getcwd(), test_labels_path)

        # self.size_of_dataset = int(self.size_of_dataset_entry.get())
        train_data_percentage = float(self.train_data_percentage_entry.get())

        # progress bar
        self.create_dataset_button["state"] = "disable"
        self.create_dataset_progress_bar_intvar = IntVar()
        self.create_dataset_progress_bar_intvar.set(0)
        self.create_dataset_progress_bar = ttk.Progressbar(self.create_dataset_tab, orient=HORIZONTAL,
                                                           length=250, mode="indeterminate",
                                                           variable=self.create_dataset_progress_bar_intvar,
                                                           maximum=100)
        self.create_dataset_progress_bar.place(relx=0.5, rely=0.775, anchor=CENTER)
        # self.create_dataset_progress_bar_label_stringvar = StringVar()
        # self.create_dataset_progress_bar_label_stringvar.set("0 %")
        # self.create_dataset_progress_bar_label = ttk.Label(self.create_dataset_tab,
        #                                                    textvariable=self.create_dataset_progress_bar_label_stringvar)
        # self.create_dataset_progress_bar_label.place(relx=0.5, rely=0.72, anchor=CENTER)
        create_dataset_thread_function = partial(main, create_dataset_now=True, root_dir=self.root_dir, fruits=fruits,
                                                 sample_time=sample_time,
                                                 sample_location=sample_location,
                                                 sample_type=sample_type,
                                                 train_spectrum_path=train_spectrum_path,
                                                 train_labels_path=train_labels_path,
                                                 test_spectrum_path=test_spectrum_path,
                                                 test_labels_path=test_labels_path,
                                                 # size_of_dataset=self.size_of_dataset,
                                                 train_data_percentage=train_data_percentage,
                                                 stretch_data=False,
                                                 create_dataset_progress_bar_intvar=None)
        self.create_dataset_thread = threading.Thread(target=create_dataset_thread_function)
        self.create_dataset_thread.start()
        self.after(50, self.update_labels_of_progress_bar, self.create_dataset_thread, self.create_dataset_progress_bar,
                   self.create_dataset_button, self.create_dataset_progress_bar_intvar, "Create Dataset",
                   "Dataset created successfully")

    def update_labels_of_progress_bar(self, thread, progress_bar, button, progress_bar_intvar, title, message):
        if thread.is_alive():
            # percent = (progress_bar_intvar.get() / progress_bar["maximum"]) * 100
            new_progress_bar_value = (progress_bar_intvar.get() % 100) + 10
            progress_bar_intvar.set(new_progress_bar_value)
            self.after(500, self.update_labels_of_progress_bar, thread, progress_bar, button, progress_bar_intvar,
                       title, message)
        else:
            messagebox.showinfo(title=title,
                                message=message)
            button["state"] = "normal"
            progress_bar.destroy()

    def design_predict_tab(self):
        self.tabs_in_predict_parent = ttk.Notebook(self.predict_tab)
        self.settings_predict_tab = ttk.Frame(self.tabs_in_predict_parent)
        self.advanced_predict_tab = ttk.Frame(self.tabs_in_predict_parent)
        self.tabs_in_predict_parent.add(self.settings_predict_tab, text="Settings")
        self.tabs_in_predict_parent.add(self.advanced_predict_tab, text="Advanced")
        self.tabs_in_predict_parent.pack(expand=1, fill="both")

        self.design_predict_settings_tab()
        self.design_predict_advanced_tab()

        # predict button
        self.predict_button = ttk.Button(self.tabs_in_predict_parent, text="Predict",
                                         command=self.predict)

        self.predict_button.place(relx=0.5, rely=0.90, anchor=CENTER)

    def design_predict_advanced_tab(self):
        # batch normalization
        self.check_button_batch_norm_in_predict_intvar = IntVar()
        self.check_button_batch_norm_in_predict_intvar.set(1)
        self.check_button_batch_norm_in_predict = ttk.Checkbutton(self.advanced_predict_tab,
                                                                  text="Batch normalization",
                                                                  variable=self.check_button_batch_norm_in_predict_intvar)
        self.check_button_batch_norm_in_predict.place(relx=0.237, rely=0.07, anchor=CENTER)
        # batch normalization tooltip
        self.check_button_batch_norm_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.check_button_batch_norm_in_predict_tooltip.image = self.tooltip_icon
        self.check_button_batch_norm_in_predict_tooltip["image"] = self.check_button_batch_norm_in_predict_tooltip.image
        self.check_button_batch_norm_in_predict_tooltip.place(relx=0.04, rely=0.07, anchor=CENTER)
        create_tooltip(widget=self.check_button_batch_norm_in_predict_tooltip, text="Check this if the model was "
                                                                                    "trained with batch normalization")

        # kernel size
        self.kernel_size_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Kernel size: ")
        self.kernel_size_in_predict_label.place(relx=0.156, rely=0.15, anchor=CENTER)
        self.kernel_size_in_predict_stringvar = StringVar()
        self.kernel_size_in_predict_stringvar.set(DEFAULTS["kernel_size"])
        self.kernel_size_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                  borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                  textvariable=self.kernel_size_in_predict_stringvar, bg="white")
        self.kernel_size_in_predict_entry.place(relx=0.685, rely=0.15, anchor=CENTER)
        # kernel size tooltip
        self.kernel_size_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.kernel_size_in_predict_tooltip.image = self.tooltip_icon
        self.kernel_size_in_predict_tooltip["image"] = self.kernel_size_in_predict_tooltip.image
        self.kernel_size_in_predict_tooltip.place(relx=0.04, rely=0.15, anchor=CENTER)
        create_tooltip(widget=self.kernel_size_in_predict_tooltip, text="Kernel size used while training the model.\n"
                                                                        "Expecting two natural numbers, surrounded with"
                                                                        " parenthesis, separated with a comma.")

        # padding
        self.padding_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Padding: ")
        self.padding_in_predict_label.place(relx=0.144, rely=0.23, anchor=CENTER)
        self.padding_in_predict_stringvar = StringVar()
        self.padding_in_predict_stringvar.set(DEFAULTS["padding"])
        self.padding_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                              borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                              textvariable=self.padding_in_predict_stringvar, bg="white")
        self.padding_in_predict_entry.place(relx=0.685, rely=0.23, anchor=CENTER)
        # padding tooltip
        self.padding_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.padding_in_predict_tooltip.image = self.tooltip_icon
        self.padding_in_predict_tooltip["image"] = self.padding_in_predict_tooltip.image
        self.padding_in_predict_tooltip.place(relx=0.04, rely=0.23, anchor=CENTER)
        create_tooltip(widget=self.padding_in_predict_tooltip, text="Padding used while training the model.\n"
                                                                    "Expecting two natural numbers, surrounded with "
                                                                    "parenthesis, separated with a comma.")

        # num channels layer 1
        self.num_channels_layer1_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Number of channels in "
                                                                                              "layer 1: ")
        self.num_channels_layer1_in_predict_label.place(relx=0.287, rely=0.31, anchor=CENTER)
        self.num_channels_layer1_in_predict_stringvar = StringVar()
        self.num_channels_layer1_in_predict_stringvar.set(DEFAULTS["num_channels_layer1"])
        self.num_channels_layer1_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                          borderwidth=0, highlightthickness=1,
                                                          highlightbackground="grey",
                                                          textvariable=self.num_channels_layer1_in_predict_stringvar,
                                                          bg="white")
        self.num_channels_layer1_in_predict_entry.place(relx=0.685, rely=0.31, anchor=CENTER)
        # num channels layer 1 tooltip
        self.num_channels_layer1_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.num_channels_layer1_in_predict_tooltip.image = self.tooltip_icon
        self.num_channels_layer1_in_predict_tooltip["image"] = self.num_channels_layer1_in_predict_tooltip.image
        self.num_channels_layer1_in_predict_tooltip.place(relx=0.04, rely=0.31, anchor=CENTER)
        create_tooltip(widget=self.num_channels_layer1_in_predict_tooltip, text="Number of channels in layer 1 used "
                                                                                "while training the model.\n"
                                                                                "Layer 1 is a convolutional layer.\n"
                                                                                "Expecting natural number.")

        # num channels layer 2
        self.num_channels_layer2_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Number of channels in "
                                                                                              "layer 2: ")
        self.num_channels_layer2_in_predict_label.place(relx=0.287, rely=0.39, anchor=CENTER)
        self.num_channels_layer2_in_predict_stringvar = StringVar()
        self.num_channels_layer2_in_predict_stringvar.set(DEFAULTS["num_channels_layer2"])
        self.num_channels_layer2_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                          borderwidth=0, highlightthickness=1,
                                                          highlightbackground="grey",
                                                          textvariable=self.num_channels_layer2_in_predict_stringvar,
                                                          bg="white")
        self.num_channels_layer2_in_predict_entry.place(relx=0.685, rely=0.39, anchor=CENTER)
        # num channels layer 2 tooltip
        self.num_channels_layer2_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.num_channels_layer2_in_predict_tooltip.image = self.tooltip_icon
        self.num_channels_layer2_in_predict_tooltip["image"] = self.num_channels_layer2_in_predict_tooltip.image
        self.num_channels_layer2_in_predict_tooltip.place(relx=0.04, rely=0.39, anchor=CENTER)
        create_tooltip(widget=self.num_channels_layer2_in_predict_tooltip, text="Number of channels in layer 2 used "
                                                                                "while training the model.\n"
                                                                                "Layer 2 is a convolutional layer.\n"
                                                                                "Expecting a natural number.")

        # amount output nodes fc1
        self.num_output_nodes_fc1_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Number of output nodes "
                                                                                               "layer 3: ")
        self.num_output_nodes_fc1_in_predict_label.place(relx=0.298, rely=0.47, anchor=CENTER)
        self.num_output_nodes_fc1_in_predict_stringvar = StringVar()
        self.num_output_nodes_fc1_in_predict_stringvar.set(DEFAULTS["fc1_amount_output_nodes"])
        self.num_output_nodes_fc1_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                           borderwidth=0, highlightthickness=1,
                                                           highlightbackground="grey",
                                                           textvariable=self.num_output_nodes_fc1_in_predict_stringvar,
                                                           bg="white")
        self.num_output_nodes_fc1_in_predict_entry.place(relx=0.685, rely=0.47, anchor=CENTER)
        # amount output nodes fc1 tooltip
        self.num_output_nodes_fc1_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.num_output_nodes_fc1_in_predict_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc1_in_predict_tooltip["image"] = self.num_output_nodes_fc1_in_predict_tooltip.image
        self.num_output_nodes_fc1_in_predict_tooltip.place(relx=0.04, rely=0.47, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc1_in_predict_tooltip, text="Number of output nodes in layer 3 "
                                                                                 "used while training the model.\n"
                                                                                 "Layer 3 is a fully connected layer.\n"
                                                                                 "Expecting a natural number.")

        # amount output nodes fc2
        self.num_output_nodes_fc2_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Number of output nodes "
                                                                                               "layer 4: ")
        self.num_output_nodes_fc2_in_predict_label.place(relx=0.298, rely=0.55, anchor=CENTER)
        self.num_output_nodes_fc2_in_predict_stringvar = StringVar()
        self.num_output_nodes_fc2_in_predict_stringvar.set(DEFAULTS["fc2_amount_output_nodes"])
        self.num_output_nodes_fc2_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                           borderwidth=0, highlightthickness=1,
                                                           highlightbackground="grey",
                                                           textvariable=self.num_output_nodes_fc2_in_predict_stringvar,
                                                           bg="white")
        self.num_output_nodes_fc2_in_predict_entry.place(relx=0.685, rely=0.55, anchor=CENTER)
        # amount output nodes fc2 tooltip
        self.num_output_nodes_fc2_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.num_output_nodes_fc2_in_predict_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc2_in_predict_tooltip["image"] = self.num_output_nodes_fc2_in_predict_tooltip.image
        self.num_output_nodes_fc2_in_predict_tooltip.place(relx=0.04, rely=0.55, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc2_in_predict_tooltip, text="Number of output nodes in layer 4 "
                                                                                 "used while training the model.\n"
                                                                                 "Layer 4 is a fully connected layer.\n"
                                                                                 "Expecting a natural number.")

        # amount output nodes fc3
        self.num_output_nodes_fc3_in_predict_label = ttk.Label(self.advanced_predict_tab, text="Number of output nodes "
                                                                                               "layer 5: ")
        self.num_output_nodes_fc3_in_predict_label.place(relx=0.298, rely=0.63, anchor=CENTER)
        self.num_output_nodes_fc3_in_predict_stringvar = StringVar()
        self.num_output_nodes_fc3_in_predict_stringvar.set(DEFAULTS["fc3_amount_output_nodes"])
        self.num_output_nodes_fc3_in_predict_entry = Entry(self.advanced_predict_tab, justify=CENTER,
                                                           borderwidth=0, highlightthickness=1,
                                                           highlightbackground="grey",
                                                           textvariable=self.num_output_nodes_fc3_in_predict_stringvar,
                                                           bg="white")
        self.num_output_nodes_fc3_in_predict_entry.place(relx=0.685, rely=0.63, anchor=CENTER)
        # amount output nodes fc3 tooltip
        self.num_output_nodes_fc3_in_predict_tooltip = ttk.Label(self.advanced_predict_tab)
        self.num_output_nodes_fc3_in_predict_tooltip.image = self.tooltip_icon
        self.num_output_nodes_fc3_in_predict_tooltip["image"] = self.num_output_nodes_fc3_in_predict_tooltip.image
        self.num_output_nodes_fc3_in_predict_tooltip.place(relx=0.04, rely=0.63, anchor=CENTER)
        create_tooltip(widget=self.num_output_nodes_fc3_in_predict_tooltip, text="Number of output nodes in layer 5 "
                                                                                 "used while training the model.\n"
                                                                                 "Layer 5 is a fully connected layer.\n"
                                                                                 "Expecting a natural number.")

    def design_predict_settings_tab(self):
        # model file
        # self.model_label_frame_in_predict = ttk.LabelFrame(self.settings_predict_tab, text="Pre-trained model file")
        # self.model_label_frame_in_predict.place(relx=0.5, rely=0.13, anchor=CENTER)
        self.model_frame_in_predict = ttk.Frame(self.settings_predict_tab, borderwidth=2, relief=SUNKEN)
        self.model_frame_in_predict.place(relx=0.5, rely=0.15, anchor=CENTER)
        self.model_file_in_predict_label = ttk.Label(self.settings_predict_tab, text="Pre-trained model file")
        self.model_file_in_predict_label.place(relx=0.5, rely=0.08, anchor=CENTER)
        self.model_file_in_predict_stringvar = StringVar()
        self.model_file_in_predict_entry = Entry(self.model_frame_in_predict,
                                                 textvariable=self.model_file_in_predict_stringvar, justify=CENTER,
                                                 borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                 background="white")
        self.model_file_in_predict_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        if os.path.exists(DEFAULTS["model_path"]):
            # self.model_path = os.path.join(os.getcwd(), DEFAULTS["model_path"])
            self.model_file_in_predict_stringvar.set(os.path.join(os.getcwd(), DEFAULTS["model_path"]))
        model_file_dialog = partial(self.file_dialog, data_or_model_file_path=MODEL_PATH)
        self.model_browse_button_in_predict = ttk.Button(self.model_frame_in_predict, text="Browse",
                                                         command=model_file_dialog)
        self.model_browse_button_in_predict.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # data to predict file
        self.predict_data_file_frame_in_predict = ttk.Frame(self.settings_predict_tab, borderwidth=2, relief=SUNKEN)
        self.predict_data_file_frame_in_predict.place(relx=0.5, rely=0.35, anchor=CENTER)
        self.predict_data_file_in_predict_label = ttk.Label(self.settings_predict_tab, text="Data to predict file")
        self.predict_data_file_in_predict_label.place(relx=0.5, rely=0.28, anchor=CENTER)
        self.predict_data_file_in_predict_stringvar = StringVar()
        self.predict_data_file_in_predict_entry = Entry(self.predict_data_file_frame_in_predict,
                                                        textvariable=self.predict_data_file_in_predict_stringvar,
                                                        justify=CENTER, borderwidth=0, highlightthickness=1,
                                                        highlightbackground="grey", background="white")
        self.predict_data_file_in_predict_entry.pack(side=RIGHT, padx=0.5, pady=1, fill=Y)
        predict_data_file_dialog = partial(self.file_dialog, data_or_model_file_path=PREDICT_DATA_PATH)
        self.predict_data_browse_button_in_predict = ttk.Button(self.predict_data_file_frame_in_predict, text="Browse",
                                                                command=predict_data_file_dialog)
        self.predict_data_browse_button_in_predict.pack(side=RIGHT, padx=0, pady=0, fill=Y)

        # fruits list
        self.fruits_in_predict_label = ttk.Label(self.settings_predict_tab,
                                                 text="Enter fruits (comma separated):")
        self.fruits_in_predict_label.place(relx=0.3, rely=0.52, anchor=CENTER)
        validate_fruit_list_cmd = (self.register(self.validate_fruit_list))  # validate command
        self.fruit_list_stringvar = StringVar()
        self.fruit_list_in_predict_entry = Entry(self.settings_predict_tab, justify=CENTER,
                                                 borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                 textvariable=self.fruit_list_stringvar, bg="white",
                                                 validate='all', validatecommand=(validate_fruit_list_cmd, '%P'))
        self.fruit_list_in_predict_entry.place(relx=0.7, rely=0.52, anchor=CENTER)
        self.validate_fruit_list(text="-1")  # hack to fill the default value

        # confidence threshold
        self.confidence_threshold_label = ttk.Label(self.settings_predict_tab, text="Confidence threshold:")
        self.confidence_threshold_label.place(relx=0.3, rely=0.68, anchor=CENTER)
        validate_confidence_threshold_cmd = (self.register(self.validate_confidence_threshold))  # validate command
        self.confidence_threshold_stringvar = StringVar()
        self.confidence_threshold_entry = Entry(self.settings_predict_tab, justify=CENTER,
                                                borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                textvariable=self.confidence_threshold_stringvar, bg="white",
                                                validate='all',
                                                validatecommand=(validate_confidence_threshold_cmd, '%P'))
        self.confidence_threshold_entry.place(relx=0.7, rely=0.68, anchor=CENTER)
        self.validate_confidence_threshold(text="-1")  # hack to fill the default value

    def flash_widget(self, widget, bg_color_to_flash="indian red", prev_bg_color="white", count=0):
        if count < 6:
            if count % 2 != 0:
                widget.configure(background=prev_bg_color)
            else:
                widget.configure(background=bg_color_to_flash)
            self.after(250, self.flash_widget, widget, bg_color_to_flash, prev_bg_color, count+1)

    def is_valid_train_inputs(self):
        illegal_inputs = []

        fruits = self.fruit_list_in_train_entry.get()
        if not fruits:
            illegal_inputs.append(self.fruit_list_in_train_entry)

        train_spectrum_path = self.train_spectrum_file_in_train_stringvar.get()
        if not train_spectrum_path:
            illegal_inputs.append(self.train_spectrum_file_in_train_entry)
        if not os.path.exists(train_spectrum_path) and self.train_spectrum_file_in_train_entry not in illegal_inputs:
            illegal_inputs.append(self.train_spectrum_file_in_train_entry)

        train_labels_path = self.train_labels_file_in_train_stringvar.get()
        if not train_labels_path:
            illegal_inputs.append(self.train_labels_file_in_train_entry)
        if not os.path.exists(train_labels_path) and self.train_labels_file_in_train_entry not in illegal_inputs:
            illegal_inputs.append(self.train_labels_file_in_train_entry)

        test_spectrum_path = self.test_spectrum_file_in_train_stringvar.get()
        if not test_spectrum_path:
            illegal_inputs.append(self.test_spectrum_file_in_train_entry)
        if not os.path.exists(test_spectrum_path) and self.test_spectrum_file_in_train_entry not in illegal_inputs:
            illegal_inputs.append(self.test_spectrum_file_in_train_entry)

        test_labels_path = self.test_labels_file_in_train_stringvar.get()
        if not test_labels_path:
            illegal_inputs.append(self.test_labels_file_in_train_entry)
        if not os.path.exists(test_labels_path) and self.test_labels_file_in_train_entry not in illegal_inputs:
            illegal_inputs.append(self.test_labels_file_in_train_entry)

        try:
            num_epochs = int(self.epoch_num_in_train_stringvar.get())
        except ValueError:
            illegal_inputs.append(self.epoch_num_in_train_entry)

        saved_model_path = self.saved_model_path_in_train_stringvar.get()
        if not saved_model_path:
            illegal_inputs.append(self.saved_model_path_in_train_entry)

        for illegal_input in illegal_inputs:
            self.flash_widget(illegal_input)

        return illegal_inputs == []

    def is_valid_create_dataset_inputs(self):
        illegal_inputs = []
        fruits = self.fruit_list_in_create_dataset_entry.get()
        if not fruits:
            illegal_inputs.append(self.fruit_list_in_create_dataset_entry)

        train_data_percentage = self.train_data_percentage_entry.get()
        if not train_data_percentage:
            illegal_inputs.append(self.train_data_percentage_entry)

        train_spectrum_path = self.train_spectrum_path_entry.get()
        if not train_spectrum_path:
            illegal_inputs.append(self.train_spectrum_path_entry)

        train_labels_path = self.train_labels_path_entry.get()
        if not train_labels_path:
            illegal_inputs.append(self.train_labels_path_entry)

        test_spectrum_path = self.test_spectrum_path_entry.get()
        if not test_spectrum_path:
            illegal_inputs.append(self.test_spectrum_path_entry)

        test_labels_path = self.test_labels_path_entry.get()
        if not test_labels_path:
            illegal_inputs.append(self.test_labels_path_entry)

        # check that train spectrum path is unique
        if train_spectrum_path == train_labels_path or train_spectrum_path == test_spectrum_path or \
           train_spectrum_path == test_labels_path:
            if self.train_spectrum_path_entry not in illegal_inputs:
                illegal_inputs.append(self.train_spectrum_path_entry)
        # check that train labels path is unique
        if train_labels_path == train_spectrum_path or train_labels_path == test_spectrum_path or \
           train_labels_path == test_labels_path:
            if self.train_labels_path_entry not in illegal_inputs:
                illegal_inputs.append(self.train_labels_path_entry)

        # check that test spectrum path is unique
        if test_spectrum_path == train_labels_path or test_spectrum_path == train_spectrum_path or \
           test_spectrum_path == test_labels_path:
            if self.test_spectrum_path_entry not in illegal_inputs:
                illegal_inputs.append(self.test_spectrum_path_entry)

        # check that test labels path is unique
        if test_labels_path == train_labels_path or test_labels_path == train_spectrum_path or \
           test_labels_path == test_spectrum_path:
            if self.test_labels_path_entry not in illegal_inputs:
                illegal_inputs.append(self.test_labels_path_entry)

        # size_of_dataset = self.size_of_dataset_entry.get()
        # if size_of_dataset:
        #     if int(size_of_dataset) % 1000 != 0:
        #         illegal_inputs.append(self.size_of_dataset_entry)
        # else:
        #     illegal_inputs.append(self.size_of_dataset_entry)
        #
        # if size_of_dataset and train_data_percentage:
        #     if (int(size_of_dataset) * float(train_data_percentage)) % 1000 != 0:
        #         illegal_inputs.append(self.train_data_percentage_entry)
        #         illegal_inputs.append(self.size_of_dataset_entry)

        try:
            rootdir = self.root_dir
        except AttributeError:
            rootdir = ""
            illegal_inputs.append(self.rootdir_in_create_dataset_label)
        if not os.path.exists(rootdir):
            illegal_inputs.append(self.rootdir_in_create_dataset_label)

        # for illegal_input in illegal_inputs:
        #     illegal_input.configure(text="Browse file")
        for illegal_input in illegal_inputs:
            self.flash_widget(illegal_input)

        return illegal_inputs == []

    def is_valid_predict_inputs(self):
        illegal_inputs = []
        fruits = self.fruit_list_in_predict_entry.get()
        if not fruits:
            illegal_inputs.append(self.fruit_list_in_predict_entry)

        confidence_threshold = self.confidence_threshold_entry.get()
        if not confidence_threshold:
            illegal_inputs.append(self.confidence_threshold_entry)

        # try:
        predict_data_path = self.predict_data_file_in_predict_entry.get()
        # except AttributeError:
        if predict_data_path == "":
            illegal_inputs.append(self.predict_data_file_in_predict_entry)
        if not os.path.exists(predict_data_path) and self.predict_data_file_in_predict_entry not in illegal_inputs:
            illegal_inputs.append(self.predict_data_file_in_predict_entry)

        # try:
        model_path = self.model_file_in_predict_stringvar.get()
            # model_path = self.model_path
        # except AttributeError:
        if model_path == "":
            illegal_inputs.append(self.model_file_in_predict_entry)
        if not os.path.exists(model_path) and self.model_file_in_predict_entry not in illegal_inputs:
            illegal_inputs.append(self.model_file_in_predict_entry)

        # for illegal_input in illegal_inputs:
        #     illegal_input.configure(text="Browse file")
        for illegal_input in illegal_inputs:
            self.flash_widget(illegal_input)

        return illegal_inputs == []

    def predict(self):
        if not self.is_valid_predict_inputs():
            return

        fruits = self.fruit_list_in_predict_entry.get().split(sep=",")
        fruits = [fruit.strip() for fruit in fruits]

        confidence_threshold = float(self.confidence_threshold_entry.get())

        kernel_size_left, kernel_size_right = self.kernel_size_in_predict_stringvar.get().split(",")
        kernel_size = (int(kernel_size_left.strip(" (")), int(kernel_size_right.strip(" )")))

        padding_left, padding_right = self.padding_in_predict_stringvar.get().split(",")
        padding = (int(padding_left.strip(" (")), int(padding_right.strip(" )")))

        num_channels_layer1 = int(self.num_channels_layer1_in_predict_stringvar.get())
        num_channels_layer2 = int(self.num_channels_layer2_in_predict_stringvar.get())

        num_output_nodes_fc1 = int(self.num_output_nodes_fc1_in_predict_stringvar.get())
        num_output_nodes_fc2 = int(self.num_output_nodes_fc2_in_predict_stringvar.get())
        num_output_nodes_fc3 = int(self.num_output_nodes_fc3_in_predict_stringvar.get())

        batch_normalization = bool(self.check_button_batch_norm_in_predict_intvar.get())

        predict_path = self.predict_data_file_in_predict_stringvar.get()
        model_path = self.model_file_in_predict_stringvar.get()

        confidence, prediction = main(predict_now=True, file_to_predict=predict_path,
                                      batch_normalization=batch_normalization, model_save_path=model_path,
                                      fruits=fruits, confidence_threshold=confidence_threshold, kernel_size=kernel_size,
                                      padding=padding, num_channels_layer1=num_channels_layer1,
                                      num_channels_layer2=num_channels_layer2,
                                      fc1_amount_output_nodes=num_output_nodes_fc1,
                                      fc2_amount_output_nodes=num_output_nodes_fc2,
                                      fc3_amount_output_node=num_output_nodes_fc3, n_components=5)

        messagebox.showinfo(title="Prediction",
                            message="Prediction: {}\tConfidence: {:.3f}%".format(prediction, confidence*100))

    def file_dialog(self, data_or_model_file_path):

        if data_or_model_file_path == MODEL_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select model file",
                                                  filetype=(("model files", "*.pth"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                # self.model_path = filename
                self.model_file_in_predict_stringvar.set(filename)

        elif data_or_model_file_path == PREDICT_DATA_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select data file",
                                                  filetype=(("data files", "*.txt"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                # self.predict_data_path = filename
                self.predict_data_file_in_predict_stringvar.set(filename)
                # self.predict_data_in_predict_label.configure(text=filename)

        elif data_or_model_file_path == ROOT_DIR:
            root_dir = filedialog.askdirectory(initialdir="./", title="Select folder containing all data set")
            if root_dir != "":  # The user didn't pick anything
                self.root_dir = root_dir
                self.rootdir_in_create_dataset_label.configure(text=root_dir)

        elif data_or_model_file_path == TRAIN_SPECTRUM_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select train spectrum file",
                                                  filetype=(("data files", "*.npy"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.train_spectrum_file_in_train_stringvar.set(filename)

        elif data_or_model_file_path == TRAIN_LABELS_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select train labels file",
                                                  filetype=(("data files", "*.npy"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.train_labels_file_in_train_stringvar.set(filename)

        elif data_or_model_file_path == TEST_SPECTRUM_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select test spectrum file",
                                                  filetype=(("data files", "*.npy"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.test_spectrum_file_in_train_stringvar.set(filename)

        elif data_or_model_file_path == TEST_LABELS_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select test labels file",
                                                  filetype=(("data files", "*.npy"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.test_labels_file_in_train_stringvar.set(filename)



    def validate_fruit_list(self, text):
        if text == "-1":
            self.fruit_list_stringvar.set(DEFAULTS["fruits"])

        # allow only letters and commas
        for c in text:
            if not c.isalpha():
                if c not in [",", " "]:
                    return False

        # don't allow two commas in a row
        prev_char = ","
        for cur_char in text:
            if prev_char == "," and cur_char == ",":
                return False
            if cur_char != " ":
                prev_char = cur_char

        return True

    def validate_confidence_threshold(self, text):
        if text == "-1":
            self.confidence_threshold_stringvar.set(DEFAULTS["confidence_threshold"])
        dot_counter = 0
        for c in text:
            if not c.isdigit():
                if c == ".":
                    dot_counter += 1
                if dot_counter > 1:  # don't allow two dots
                    return False
                if c != ".":
                    return False
        if text and float(text) > 1:  # only numbers between 0-1
            return False
        return True

    def validate_train_data_percentage(self, text):
        if text == "-1":
            self.train_data_percentage_stringvar.set(DEFAULTS["train_data_percentage"])
        dot_counter = 0
        for c in text:
            if not c.isdigit():
                if c == ".":
                    dot_counter += 1
                if dot_counter > 1:  # don't allow two dots
                    return False
                if c != ".":
                    return False
        if text and float(text) > 1:  # only numbers between 0-1
            return False
        return True

    def validate_size_of_dataset(self, text):
        if text == "-1":
            self.size_of_dataset_stringvar.set(DEFAULTS["dataset_size"])
        for c in text:
            if not c.isdigit():
                return False
        return True


if __name__ == '__main__':
    gui = FinalProjectGui()
    gui.mainloop()
