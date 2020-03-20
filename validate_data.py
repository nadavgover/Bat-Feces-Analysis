import os
import numpy as np
import collections
import itertools
from common import read_data

InvalidFile = collections.namedtuple('InvalidFile', 'file_path reason')
Data = collections.namedtuple('Data', 'file_path data')

FRUITS = ["apple", "banana", "mix"]
SAMPLE_LOCATIONS = ["oral", "anal"]
SAMPLE_TIMES = ["after 5", "after 8", "before"]


def __is_file_saved_in_correct_directory_hierarchy(file_path):
    try:
        root, name, fruit, location, time, filename = r"{}".format(file_path).split("\\")
        if fruit.lower() not in FRUITS:
            return False
        if location.lower() not in SAMPLE_LOCATIONS:
            return False
        if time.lower() not in SAMPLE_TIMES:
            return False
    except ValueError:
        return False

    return True


def __is_filename_correct_format(filename):
    if len(filename) != 16:
        return False
    if not filename.startswith("YY20-"):
        return False
    if not all([char.isdigit() for char in filename[5:8]]):  # those 3 characters have to be digits
        return False
    if not (filename.endswith(" pos.txt") or filename.endswith(" neg.txt")):
        return False
    return True


def __is_file_empty(filename):
    return os.path.getsize(filename) == 0


def __get_identical_files(data_of_all_files):
    identical_files = []
    for file1, file2 in itertools.combinations(data_of_all_files, 2):
        full_path1, data1 = file1.file_path, file1.data
        full_path2, data2 = file2.file_path, file2.data
        files_are_identical = np.array_equal(data1, data2)
        if files_are_identical:
            identical_files.extend([InvalidFile(file_path=full_path1, reason="Identical to {}".format(full_path2)),
                                    InvalidFile(file_path=full_path2, reason="Identical to {}".format(full_path1))])

    return identical_files


def get_valid_and_invalid_files(root_dir="YOMIRAN", validate_hierarchy=True,
                                validate_filename_format=True, validate_empty_file=True):
    """Return (valid_files, invalid_files)
    Validation criteria:
        - empty file
        - format of file name
        - identical files (same content)
        - folder hierarchy of file"""
    invalid_files = []
    valid_files = []
    data_of_all_files = []  # in format of [Data1, Data2...] where Data is a named tuple
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            file_is_valid = True
            full_path = os.path.join(root, file)
            if validate_hierarchy and not __is_file_saved_in_correct_directory_hierarchy(file_path=full_path):
                invalid_files.append(InvalidFile(file_path=full_path, reason="Incorrect folder hierarchy"))
                file_is_valid = False
            #     continue
            if validate_filename_format and not __is_filename_correct_format(filename=file):
                invalid_files.append(InvalidFile(file_path=full_path, reason="Incorrect file name"))
                file_is_valid = False
                # continue
            if validate_empty_file and __is_file_empty(filename=full_path):
                invalid_files.append(InvalidFile(file_path=full_path, reason="Empty file"))
                file_is_valid = False
                # continue
            if file_is_valid:
                valid_files.append(full_path)

            # for future check if we have identical files
            data = read_data(filename=full_path)
            data_numpy = np.vstack((np.array(data["x"]), np.array(data["y"])))
            data_of_all_files.append(Data(file_path=full_path, data=data_numpy))

    identical_files = __get_identical_files(data_of_all_files)
    invalid_files.extend(identical_files)

    # filtering the identical files from the valid files
    identical_files_paths = [identical_file.file_path for identical_file in identical_files]
    valid_files = list(filter(lambda filename: filename not in identical_files_paths, valid_files))
    return valid_files, invalid_files


if __name__ == '__main__':
    valid_files, invalid_files = get_valid_and_invalid_files(root_dir="YOMIRAN1", validate_hierarchy=True,
                                                             validate_empty_file=True, validate_filename_format=True)
    save_to_file = False
    if not invalid_files:
        print("All files are valid")
    else:
        print("Number of valid files: {}".format(len(valid_files)))
        print("Number of invalid files: {}".format(len(invalid_files)))
        if save_to_file:
            with open("valid_files.txt", "w") as valid_files_writer:
                for valid_file in valid_files:
                    valid_files_writer.writelines("Valid file: {}\n".format(valid_file))
            with open("invalid_files.txt", "w") as invalid_files_writer:
                for invalid_file in invalid_files:
                    invalid_files_writer.writelines("Invalid file: {}\t\t Reason: {}\n"
                                                    .format(invalid_file.file_path, invalid_file.reason))
