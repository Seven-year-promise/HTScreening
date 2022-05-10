import numpy as np
import csv
from scipy import fftpack
import matplotlib.pyplot as plt
"""
extract feature based on
[0:187) before the first light
[187:363) after the first light
[363:) after the second light
"""
PHASES = [0, 187, 207, 227, 267, 307, 363, 383, 403, 443, 483, -1]

def feature_extract(data):
    f = 10  # frequency
    Fs = 100  # sampling rate



    length = len(data)

    t = np.arange(0, length)
    data = np.array(data)
    y_fft = fftpack.fft(data)
    # Plot data
    n = length
    fr = Fs / 2 * np.linspace(0, 1, n // 2)
    y_m = 2 / n * abs(y_fft[0:np.size(fr)])
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].plot(t, data)  # plot time series
    ax[1].stem(fr, y_m)  # plot freq domain
    plt.tight_layout()
    plt.show()

    return y_m.tolist()

def extract_feature_to(data_path, save_path):
    all_data_feature = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            one_data = [float(i) for i in l[1:-2]]
            feature = feature_extract(one_data)
            feature_data = [l[0]] + feature + l[-2:]
            all_data_feature.append(feature_data)
    with open(save_path + "all_compounds_FT_feature_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_data_feature)

if __name__ == "__main__":
    data_path = "./data/cleaned/all_compounds_ori_fish_with_action.csv"
    save_path = "./data/featured/"
    extract_feature_to(data_path, save_path)