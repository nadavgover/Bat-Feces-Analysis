from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
import os

from main import main

MODEL_PATH = 0
PREDICT_DATA_PATH = 1
ROOT_DIR = 2
DEFAULTS = {"model_path": "model.pth", "fruits": "apple, banana, mix", "kernel_size": "(2, 5)", "padding": "(1, 5)",
            "data_width": "2100", "number_of_channels_in_layer 1": "3", "number_of_channels_in_layer_2": "6",
            "confidence_threshold": "0.7", "root_dir": "YOMIRAN", "sample_times": ["after 5", "after 8", "before",
                                                                                   "after 5, after 8",
                                                                                   "after 5, before",
                                                                                   "after 8, before", "all"],
            "sample_locations": ["anal", "oral", "all"]}


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

        main(create_dataset_now=True, root_dir=self.root_dir, fruits=fruits, sample_time=sample_time,
             sample_location=sample_location)

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
        self.model_label_frame_in_predict = ttk.LabelFrame(self.settings_predict_tab, text="Pre-trained model file")
        self.model_label_frame_in_predict.place(relx=0.5, rely=0.13, anchor=CENTER)
        self.model_file_in_predict_label = ttk.Label(self.model_label_frame_in_predict, text="")
        self.model_file_in_predict_label.grid(column=1, row=2)
        if os.path.exists(DEFAULTS["model_path"]):
            self.model_path = os.path.join(os.getcwd(), DEFAULTS["model_path"])
        #     self.model_file_in_predict_label.grid(column=1, row=2)
            self.model_file_in_predict_label.configure(text=self.model_path)
        model_file_dialog = partial(self.file_dialog, data_or_model_file_path=MODEL_PATH)
        self.model_browse_button_in_predict = ttk.Button(self.model_label_frame_in_predict, text="Browse",
                                                         command=model_file_dialog)
        self.model_browse_button_in_predict.grid(column=1, row=1)

        # fruits list
        self.fruits_in_predict_label_frame = ttk.LabelFrame(self.settings_predict_tab, text="Fruits")
        self.fruits_in_predict_label_frame.place(relx=0.5, rely=0.35, anchor=CENTER)
        self.fruits_in_predict_label = ttk.Label(self.fruits_in_predict_label_frame,
                                                 text="Please enter fruits (comma separated)")
        self.fruits_in_predict_label.grid(column=1, row=1)
        validate_fruit_list_cmd = (self.register(self.validate_fruit_list))  # validate command
        self.fruit_list_stringvar = StringVar()
        self.fruit_list_in_predict_entry = Entry(self.fruits_in_predict_label_frame, justify=CENTER,
                                                 borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                 textvariable=self.fruit_list_stringvar, bg="white",
                                                 validate='all', validatecommand=(validate_fruit_list_cmd, '%P'))
        self.fruit_list_in_predict_entry.grid(column=1, row=2)
        self.validate_fruit_list(text="-1")  # hack to fill the default value

        # data to predict file
        self.predict_data_label_frame_in_predict = ttk.LabelFrame(self.settings_predict_tab,
                                                                  text="Data to predict file")
        self.predict_data_label_frame_in_predict.place(relx=0.5, rely=0.57, anchor=CENTER)
        self.predict_data_in_predict_label = ttk.Label(self.predict_data_label_frame_in_predict, text="")
        self.predict_data_in_predict_label.grid(column=1, row=2)
        predict_data_file_dialog = partial(self.file_dialog, data_or_model_file_path=PREDICT_DATA_PATH)
        self.predict_data_browse_button_in_predict = ttk.Button(self.predict_data_label_frame_in_predict, text="Browse",
                                                                command=predict_data_file_dialog)
        self.predict_data_browse_button_in_predict.grid(column=1, row=1)

        # confidence threshold
        self.confidence_threshold_label = ttk.Label(self.settings_predict_tab, text="Confidence threshold:")
        self.confidence_threshold_label.place(relx=0.3, rely=0.75, anchor=CENTER)
        validate_confidence_threshold_cmd = (self.register(self.validate_confidence_threshold))  # validate command
        self.confidence_threshold_stringvar = StringVar()
        self.confidence_threshold_entry = Entry(self.settings_predict_tab, justify=CENTER,
                                                borderwidth=0, highlightthickness=1, highlightbackground="grey",
                                                textvariable=self.confidence_threshold_stringvar, bg="white",
                                                validate='all',
                                                validatecommand=(validate_confidence_threshold_cmd, '%P'))
        self.confidence_threshold_entry.place(relx=0.7, rely=0.75, anchor=CENTER)
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

        try:
            rootdir = self.root_dir
        except AttributeError:
            rootdir = ""
            illegal_inputs.append(self.rootdir_in_create_dataset_label)
        if not os.path.exists(rootdir):
            illegal_inputs.append(self.rootdir_in_create_dataset_label)

        for illegal_input in illegal_inputs:
            illegal_input.configure(text="Browse file")
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

        try:
            predict_data_path = self.predict_data_path
        except AttributeError:
            predict_data_path = ""
            illegal_inputs.append(self.predict_data_in_predict_label)
        if not os.path.exists(predict_data_path):
            illegal_inputs.append(self.predict_data_in_predict_label)

        try:
            model_path = self.model_path
        except AttributeError:
            model_path = ""
            illegal_inputs.append(self.model_file_in_predict_label)
        if not os.path.exists(model_path):
            illegal_inputs.append(self.model_file_in_predict_label)

        for illegal_input in illegal_inputs:
            illegal_input.configure(text="Browse file")
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

        confidence, prediction = main(predict_now=True, file_to_predict=self.predict_data_path,
                                      model_save_path=self.model_path, fruits=fruits,
                                      confidence_threshold=confidence_threshold, kernel_size=kernel_size,
                                      padding=padding)

        messagebox.showinfo(title="Prediction",
                            message="Prediction: {}\tConfidence: {:.3f}%".format(prediction, confidence*100))

    def file_dialog(self, data_or_model_file_path):

        if data_or_model_file_path == MODEL_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select model file",
                                                  filetype=(("pth files", "*.pth"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.model_path = filename
                # self.model_file_in_predict_label.grid(column=1, row=2)
                self.model_file_in_predict_label.configure(text=filename)

        elif data_or_model_file_path == PREDICT_DATA_PATH:
            filename = filedialog.askopenfilename(initialdir="./", title="Select data file",
                                                  filetype=(("txt files", "*.txt"), ("all files", "*.*")))
            if filename != "":  # The user didn't pick anything
                self.predict_data_path = filename
                # self.predict_data_in_predict_label.grid(column=1, row=3)
                self.predict_data_in_predict_label.configure(text=filename)

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


if __name__ == '__main__':
    gui = FinalProjectGui()
    gui.mainloop()
