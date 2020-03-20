def read_data(filename):
    data = {}
    x_axis = []
    y_axis = []
    with open(filename, "r") as data_file:
        for line in data_file:
            try:
                x_data, y_data = line.rstrip('\n').split('\t')
            except ValueError:  # end of file
                continue
            x_axis.append(float(x_data))
            y_axis.append(float(y_data))

    data["x"] = x_axis
    data["y"] = y_axis
    return data
