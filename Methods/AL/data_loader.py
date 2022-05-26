import csv
import numpy as np
import matplotlib.pyplot as plt

def load_cleaned_data(path):
    all_data = []
    all_comps = []
    all_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):


            one_data = [float(i) for i in l[1:-2]]
            if len(one_data) != 539:
                print("oops!")
                continue
            all_comps.append(l[0])
            #print(len(one_data))
            all_data.append(one_data)
            all_labels.append(int(l[-1]))
    #print(all_data)
    all_data = np.array(all_data)
    all_comps = np.array(all_comps)
    all_labels = np.array(all_labels)
    print(all_data.shape)
    return all_data, all_comps, all_labels

def generate_training_set(all_data, all_labels):
    selected_x = []
    selected_y = []
    for c in range(12):
        selected_y.append(c)
        class_data = all_data[all_labels==c, :]
        print(class_data.shape)
        selected_x.append(np.median(class_data, axis=1))

    plt.plot(selected_x)
    plt.show()
    return np.array(selected_x), np.array(selected_y), all_data