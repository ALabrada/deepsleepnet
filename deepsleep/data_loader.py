import os

import numpy as np

from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_oversample

import re
import csv


class NonSeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_csv_file(selfs, csv_file):
        """Load data and labels from a csv file."""
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = []
            labels = []
            for row in reader:
                try:
                    labels.append(int(row[0]))
                    features = [float(x) for x in row[1:]]
                    data.append(np.array(features, dtype=np.float))
                except:
                    pass

            data = np.vstack(data)
            mean = np.mean(data, axis=0)
            stdev = np.std(data, axis=0)
            data = (data - mean) / (3 * stdev)
            data = (data + 1) / 2
            data = np.maximum(0, np.minimum(data, 1))

            labels = np.array(labels, dtype=np.int32)
            return data, labels

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_csv_list_files(self, csv_files):
        """Load data and labels from list of csv files."""
        data = []
        labels = []
        for csv_f in csv_files:
            print("Loading {} ...".format(csv_f))
            tmp_data, tmp_labels = self._load_csv_file(csv_f)
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        data = data[:, :, np.newaxis]
        return data, labels

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_cv_data(self, list_files, csv=False):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        if csv:
            # Load a csv file
            print("Load training set:")
            data_train, label_train = self._load_csv_list_files(train_files)
            print(" ")
            print("Load validation set:")
            data_val, label_val = self._load_csv_list_files(val_files[self.fold_idx])
            print(" ")
        else:
            # Load a npz file
            print("Load training set:")
            data_train, label_train = self._load_npz_list_files(train_files)
            print(" ")
            print("Load validation set:")
            data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
            print(" ")

        # Reshape the data to match the input of the model - conv2d
        # data_train = np.squeeze(data_train)
        # data_val = np.squeeze(data_val)
        data_train = data_train[:, :, :, np.newaxis]
        data_val = data_val[:, :, :, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        csvfiles = []
        for idx, f in enumerate(allfiles):
            if ".csv" in f:
                csvfiles.append(os.path.join(self.data_dir, f))
        csvfiles.sort()

        # if n_files is not None:
        #     npzfiles = npzfiles[:n_files]
        #
        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))
        #
        # if len(subject_files) == 0:
        #     for idx, f in enumerate(allfiles):
        #         if self.fold_idx < 10:
        #             pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
        #         else:
        #             pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
        #         if pattern.match(f):
        #             subject_files.append(os.path.join(self.data_dir, f))
        #
        # train_files = list(set(npzfiles) - set(subject_files))
        # train_files.sort()
        # subject_files.sort()
        #
        # # Load training and validation sets
        # print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        # print("Load training set:")
        # data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        # print(" ")
        # print("Load validation set:")
        # data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        # print(" ")

        # # Reshape the data to match the input of the model - conv2d
        # data_train = np.squeeze(data_train)
        # data_val = np.squeeze(data_val)
        # data_train = data_train[:, :, :, np.newaxis]
        # data_val = data_val[:, :, :, np.newaxis]
        #
        # # Casting
        # data_train = data_train.astype(np.float32)
        # label_train = label_train.astype(np.int32)
        # data_val = data_val.astype(np.float32)
        # label_val = label_val.astype(np.int32)

        if len(npzfiles) > 0:
            data_train, label_train, data_val, label_val = self._load_cv_data(npzfiles, csv=False)
        else:
            data_train, label_train, data_val, label_val = self._load_cv_data(csvfiles, csv=True)

        print("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print(" ")
        print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val)
        print(" ")

        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print(" ")

        return x_train, y_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()
        csvfiles = []
        for idx, f in enumerate(allfiles):
            if ".csv" in f:
                csvfiles.append(os.path.join(self.data_dir, f))
        csvfiles.sort()

        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))
        # subject_files.sort()
        #
        # print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        #
        # print("Load validation set:")
        # data_val, label_val = self._load_npz_list_files(subject_files)

        # # Reshape the data to match the input of the model
        # data_val = np.squeeze(data_val)
        # data_val = data_val[:, :, np.newaxis, np.newaxis]
        #
        # # Casting
        # data_val = data_val.astype(np.float32)
        # label_val = label_val.astype(np.int32)

        if len(npzfiles) > 0:
            data_train, label_train, data_val, label_val = self._load_cv_data(npzfiles, csv=False)
        else:
            data_train, label_train, data_val, label_val = self._load_cv_data(csvfiles, csv=True)

        return data_val, label_val


class SeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_csv_file(selfs, csv_file):
        """Load data and labels from a csv file."""
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = []
            labels = []
            for row in reader:
                try:
                    labels.append(int(row[0]))
                    features = [float(x) for x in row[1:]]
                    data.append(np.array(features, dtype=np.float))
                except:
                    pass

            data = np.vstack(data)
            mean = np.mean(data, axis=0)
            stdev = np.std(data, axis=0)
            data = (data - mean) / (3 * stdev)
            data = (data + 1) / 2
            data = np.maximum(0, np.minimum(data, 1))

            labels = np.array(labels, dtype=np.int32)
            return data, labels

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_csv_list_files(self, csv_files):
        """Load data and labels from list of csv files."""
        data = []
        labels = []
        for csv_f in csv_files:
            print("Loading {} ...".format(csv_f))
            tmp_data, tmp_labels = self._load_csv_file(csv_f)

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, :, np.newaxis]
            
            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_cv_data(self, list_files, csv=False):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        if csv:
            # Load a csv file
            print("Load training set:")
            data_train, label_train = self._load_csv_list_files(train_files)
            print(" ")
            print("Load validation set:")
            data_val, label_val = self._load_csv_list_files(val_files[self.fold_idx])
            print(" ")
        else:
            # Load a npz file
            print("Load training set:")
            data_train, label_train = self._load_npz_list_files(train_files)
            print(" ")
            print("Load validation set:")
            data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
            print(" ")

        return data_train, label_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()
        csvfiles = []
        for idx, f in enumerate(allfiles):
            if ".csv" in f:
                csvfiles.append(os.path.join(self.data_dir, f))
        csvfiles.sort()

        # # Files for validation sets
        # val_files = np.array_split(npzfiles, self.n_folds)
        # val_files = val_files[self.fold_idx]
        #
        # print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        #
        # print("Load validation set:")
        # data_val, label_val = self._load_npz_list_files(val_files)

        if len(npzfiles) > 0:
            data_train, label_train, data_val, label_val = self._load_cv_data(npzfiles, csv=False)
        else:
            data_train, label_train, data_val, label_val = self._load_cv_data(csvfiles, csv=True)

        return data_val, label_val

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()
        csvfiles = []
        for idx, f in enumerate(allfiles):
            if ".csv" in f:
                csvfiles.append(os.path.join(self.data_dir, f))
        csvfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))
        #
        # train_files = list(set(npzfiles) - set(subject_files))
        # train_files.sort()
        # subject_files.sort()
        #
        # # Load training and validation sets
        # print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        # print("Load training set:")
        # data_train, label_train = self._load_npz_list_files(train_files)
        # print(" ")
        # print("Load validation set:")
        # data_val, label_val = self._load_npz_list_files(subject_files)
        # print(" ")

        if len(npzfiles) > 0:
            data_train, label_train, data_val, label_val = self._load_cv_data(npzfiles, csv=False)
        else:
            data_train, label_train, data_val, label_val = self._load_cv_data(csvfiles, csv=True)

        print("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            print(d.shape)
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        print_n_samples_each_class(np.hstack(label_train))
        print(" ")
        print("Validation set: n_subjects={}".format(len(data_val)))
        n_valid_examples = 0
        for d in data_val:
            print(d.shape)
            n_valid_examples += d.shape[0]
        print("Number of examples = {}".format(n_valid_examples))
        print_n_samples_each_class(np.hstack(label_val))
        print(" ")

        return data_train, label_train, data_val, label_val

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        subject_files = []
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, f))

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print("Loading {} ...".format(npz_f))
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, :, np.newaxis]
                
                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print("Load data from: {}".format(subject_files))
        data, labels = load_npz_list_files(subject_files)

        return data, labels
