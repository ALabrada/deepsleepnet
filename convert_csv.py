import argparse
import os
import re

import numpy as np
import csv


def convert_file(csv_file, npz_file):
    data = []
    labels = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            try:
                labels.append(int(row[0]))
                features = [float(x) for x in row[1:]]
                data.append(np.array(features, dtype=np.float))
            except:
                pass

    data = np.vstack(data)
    baseline = np.mean(data, axis=0)
    var = 3 * np.std(data, axis=0)
    data = data - baseline
    out_of_range = var > 1
    data[:, out_of_range] = data[:, out_of_range] / var[out_of_range]
    data = (data + 1) / 2
    data = np.maximum(0, np.minimum(data, 1))
    data = data[:, :, np.newaxis]

    labels = np.array(labels, dtype=np.int32)

    save_dict = {
        "x": data,
        "y": labels,
        "fs": None,
    }
    np.savez(npz_file, **save_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory where to load csv files")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="Directory where to write npz files")
    args = parser.parse_args()

    allfiles = os.listdir(args.data_dir)
    for idx, f in enumerate(allfiles):
        if ".csv" in f:
            in_path = os.path.join(args.data_dir, f)
            out_path = os.path.join(args.out_dir, f.replace(".csv", ".npz"))
            print("Converting file {}".format(f))
            convert_file(in_path, out_path)
        elif ".npz" in f:
            in_path = os.path.join(args.data_dir, f)
            file = np.load(in_path)
            data = {"x": file["x"], "y": file["y"], "fs": 1}
            np.savez(in_path, **data)


if __name__ == "__main__":
    main()