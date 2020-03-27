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
DEFAULTS = {"model_path": "model.pth", "fruits": "apple, banana, mix", "kernel_size": "(2, 2)", "padding": "(1, 1)",
            "data_width": "2100", "number_of_channels_in_layer 1": "3", "number_of_channels_in_layer_2": "6",
            "confidence_threshold": "0.7", "root_dir": "YOMIRAN", "sample_times": ["after 5", "after 8", "before",
                                                                                   "after 5, after 8",
                                                                                   "after 5, before",
                                                                                   "after 8, before", "all"],
            "sample_locations": ["anal", "oral", "all"], "train_spectrum_path": "train_spectrum",
            "test_spectrum_path": "test_spectrum", "train_labels_path": "train_labels",
            "test_labels_path": "test_labels", "dataset_size": "60000", "train_data_percentage": "0.8",
            "dataset_folder_name": "dataset"}


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

        self.kernel_size = DEFAULTS["kernel_size"]
        self.padding = DEFAULTS["padding"]
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
        self.sample_time_in_create_dataset_label.place(relx=0.28, rely=0.57, anchor=CENTER)
        self.sample_time_combo_box = ttk.Combobox(self.settings_create_dataset_tab, state="readonly",
                                                  values=DEFAULTS["sample_times"])
        self.sample_time_combo_box.set(DEFAULTS["sample_times"][0])
        self.sample_time_combo_box.place(relx=0.28, rely=0.63, anchor=CENTER)

        # sample location
        self.sample_location_in_create_dataset_label = ttk.Label(self.settings_create_dataset_tab,
                                                                 text="Sample location")
        self.sample_location_in_create_dataset_label.place(relx=0.72, rely=0.57, anchor=CENTER)
        self.sample_location_combo_box = ttk.Combobox(self.settings_create_dataset_tab, state="readonly",
                                                      values=DEFAULTS["sample_locations"])
        self.sample_location_combo_box.set(DEFAULTS["sample_locations"][0])
        self.sample_location_combo_box.place(relx=0.72, rely=0.63, anchor=CENTER)

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

        # size of dataset
        self.size_of_dataset_label = ttk.Label(self.advanced_create_dataset_tab, text="Size of dataset: ")
        self.size_of_dataset_label.place(relx=0.185, rely=0.47, anchor=CENTER)
        self.size_of_dataset_stringvar = StringVar()
        validate_size_of_dataset_cmd = (self.register(self.validate_size_of_dataset))  # validate command
        self.size_of_dataset_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                           borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                           textvariable=self.size_of_dataset_stringvar, bg="white",
                                           validate='all',
                                           validatecommand=(validate_size_of_dataset_cmd, '%P'))
        self.size_of_dataset_entry.place(relx=0.58, rely=0.47, anchor=CENTER)
        self.validate_size_of_dataset(text="-1")
        # size of dataset tooltip
        self.size_of_dataset_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.size_of_dataset_tooltip.image = self.tooltip_icon
        self.size_of_dataset_tooltip["image"] = self.size_of_dataset_tooltip.image
        self.size_of_dataset_tooltip.place(relx=0.04, rely=0.47, anchor=CENTER)
        create_tooltip(widget=self.size_of_dataset_tooltip, text="Dataset size.\nMust be divisible by a thousand.")

        # train data percentage
        self.train_data_percentage_label = ttk.Label(self.advanced_create_dataset_tab, text="train data percentage : ")
        self.train_data_percentage_label.place(relx=0.24, rely=0.55, anchor=CENTER)
        self.train_data_percentage_stringvar = StringVar()
        validate_train_data_percentage_cmd = (self.register(self.validate_train_data_percentage))  # validate command
        self.train_data_percentage_entry = Entry(self.advanced_create_dataset_tab, justify=CENTER,
                                                 borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                 textvariable=self.train_data_percentage_stringvar, bg="white",
                                                 validate='all',
                                                 validatecommand=(validate_train_data_percentage_cmd, '%P'))
        self.train_data_percentage_entry.place(relx=0.58, rely=0.55, anchor=CENTER)
        self.validate_train_data_percentage(text="-1")  # hack to get the default value
        # train data percentage  tooltip
        self.train_data_percentage_tooltip = ttk.Label(self.advanced_create_dataset_tab)
        self.train_data_percentage_tooltip.image = self.tooltip_icon
        self.train_data_percentage_tooltip["image"] = self.train_data_percentage_tooltip.image
        self.train_data_percentage_tooltip.place(relx=0.04, rely=0.55, anchor=CENTER)
        create_tooltip(widget=self.train_data_percentage_tooltip, text="Determines how much out of the dataset will be "
                                                                       "the training set in percent %\n"
                                                                       "The rest of the data will be part of the "
                                                                       "test set.\n"
                                                                       "The size of the dataset times the train data "
                                                                       "percentage must be divisible by a thousand.")

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

        self.size_of_dataset = int(self.size_of_dataset_entry.get())
        train_data_percentage = float(self.train_data_percentage_entry.get())

        # progress bar
        self.create_dataset_button["state"] = "disable"
        self.create_dataset_progress_bar_intvar = IntVar()
        self.create_dataset_progress_bar_intvar.set(0)
        self.create_dataset_progress_bar = ttk.Progressbar(self.create_dataset_tab, orient=HORIZONTAL,
                                                           length=250, mode="determinate",
                                                           variable=self.create_dataset_progress_bar_intvar,
                                                           maximum=self.size_of_dataset)
        self.create_dataset_progress_bar.place(relx=0.5, rely=0.775, anchor=CENTER)
        self.create_dataset_progress_bar_label_stringvar = StringVar()
        self.create_dataset_progress_bar_label_stringvar.set("0 %")
        self.create_dataset_progress_bar_label = ttk.Label(self.create_dataset_tab,
                                                           textvariable=self.create_dataset_progress_bar_label_stringvar)
        self.create_dataset_progress_bar_label.place(relx=0.5, rely=0.72, anchor=CENTER)
        create_dataset_thread_function = partial(main, create_dataset_now=True, root_dir=self.root_dir, fruits=fruits, sample_time=sample_time,
                                                 sample_location=sample_location,
                                                 train_spectrum_path=train_spectrum_path,
                                                 train_labels_path=train_labels_path,
                                                 test_spectrum_path=test_spectrum_path,
                                                 test_labels_path=test_labels_path,
                                                 size_of_dataset=self.size_of_dataset,
                                                 train_data_percentage=train_data_percentage,
                                                 create_dataset_progress_bar_intvar=self.create_dataset_progress_bar_intvar)
        self.create_dataset_thread = threading.Thread(target=create_dataset_thread_function)
        self.create_dataset_thread.start()
        self.after(50, self.update_labels_of_progress_bar, self.create_dataset_thread, self.create_dataset_progress_bar,
                   self.create_dataset_button, self.create_dataset_progress_bar_label,
                   self.create_dataset_progress_bar_label_stringvar, self.create_dataset_progress_bar_intvar)

    def update_labels_of_progress_bar(self, thread, progress_bar, button, label, label_stringvar, progress_bar_intvar):
        if thread.is_alive():
            percent = (progress_bar_intvar.get() / progress_bar["maximum"]) * 100
            label_stringvar.set("{} %".format(int(percent)))
            self.after(500, self.update_labels_of_progress_bar, thread, progress_bar, button, label, label_stringvar,
                       progress_bar_intvar)
        else:
            label_stringvar.set("100 %")
            messagebox.showinfo(title="Create Dataset",
                                message="Dataset created successfully")
            button["state"] = "normal"
            progress_bar.destroy()
            label.destroy()

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
        pass

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

        size_of_dataset = self.size_of_dataset_entry.get()
        if size_of_dataset:
            if int(size_of_dataset) % 1000 != 0:
                illegal_inputs.append(self.size_of_dataset_entry)
        else:
            illegal_inputs.append(self.size_of_dataset_entry)

        if size_of_dataset and train_data_percentage:
            if (int(size_of_dataset) * float(train_data_percentage)) % 1000 != 0:
                illegal_inputs.append(self.train_data_percentage_entry)
                illegal_inputs.append(self.size_of_dataset_entry)

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

        kernel_size_left, kernel_size_right = self.kernel_size.split(",")
        kernel_size = (int(kernel_size_left.strip(" (")), int(kernel_size_right.strip(" )")))

        padding_left, padding_right = self.padding.split(",")
        padding = (int(padding_left.strip(" (")), int(padding_right.strip(" )")))
        predict_path = self.predict_data_file_in_predict_stringvar.get()
        model_path = self.model_file_in_predict_stringvar.get()

        confidence, prediction = main(predict_now=True, file_to_predict=predict_path,
                                      model_save_path=model_path, fruits=fruits,
                                      confidence_threshold=confidence_threshold, kernel_size=kernel_size,
                                      padding=padding)

        messagebox.showinfo(title="Prediction",
                            message="Prediction: {}\tConfidence: {:.3f}%".format(prediction, confidence*100))

    def file_dialog(self, data_or_model_file_path):

        if data_or_model_file_path == MODEL_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select model file",
                                                  filetype=(("pth files", "*.pth"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                # self.model_path = filename
                self.model_file_in_predict_stringvar.set(filename)

        elif data_or_model_file_path == PREDICT_DATA_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select data file",
                                                  filetype=(("txt files", "*.txt"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                # self.predict_data_path = filename
                self.predict_data_file_in_predict_stringvar.set(filename)
                # self.predict_data_in_predict_label.configure(text=filename)

        elif data_or_model_file_path == ROOT_DIR:
            root_dir = filedialog.askdirectory(initialdir="./", title="Select folder containing all data set")
            if root_dir != "":  # The user didn't pick anything
                self.root_dir = root_dir
                self.rootdir_in_create_dataset_label.configure(text=root_dir)

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
