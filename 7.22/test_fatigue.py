import csv
import collections
import os
from sklearn import svm
import joblib
from sklearn import svm
from sklearn.datasets import load_iris


_DATA_DIR = "new_features_gram"
_NUM_FEATURES = 12


def load_data():
    results = collections.defaultdict(dict)
    for file_name in os.listdir(_DATA_DIR):
        parts = file_name.split("_", 3)
        subject = parts[1]
        test = parts[2]
        raw_data = []
        with open(os.path.join(_DATA_DIR, file_name), newline="") as f:
            reader = csv.reader(f)
            next(reader, None) #跳過第一行名稱
            for row in reader:
                raw_data.append([float(i) for i in row])
        results[test][subject] = raw_data

    return results


def generate_baseline_vector(raw_data) -> dict:
    """Use the average of pre data point as baseline"""
    baseline_vector = {}
    for subject in raw_data:
        n_samples = int(len(raw_data[subject]) / 2)
        start = int(len(raw_data[subject]) / 4)
        vector = [0] * _NUM_FEATURES
        for i in range(start, start + n_samples):
            for k in range(_NUM_FEATURES):
                vector[k] += raw_data[subject][i][k]
        for k in range(_NUM_FEATURES):
            vector[k] /= n_samples
        baseline_vector[subject] = vector
    return baseline_vector


def compute_result(clf, data, labels, show=False):
    results = clf.predict(data)
    total = 0
    correct = 0
    for prediction, label in zip(results, labels):
        if prediction == label:
            correct += 1
        total += 1
        if show:
            print(label, prediction)
    print("Pass rate is %f" % (correct / total))


def generate_samples(data, baseline_vector, subjects):
    all_samples = []
    all_labels = []
    mapping = {"pre": -1, "post": 1}
    for subject in subjects:
        for test, label in mapping.items():
            samples = data[test][subject]
            for row in samples:
                adjusted = [0] * _NUM_FEATURES
                for k in range(_NUM_FEATURES):
                    adjusted[k] = row[k] - baseline_vector[subject][k]
                all_samples.append(adjusted)

            all_labels.extend([label] * (len(all_samples) - len(all_labels)))

    return all_samples, all_labels


def main():
    data = load_data()
    baseline_vector = generate_baseline_vector(data["pre"])

    C = 1e-5
    for i in range(1, 13):
        print(i)
        test_subjects = [f"{i}"]
        train_subjects = [f"{j}" for j in range(1, 12) if j != i]
        train_samples, train_labels = generate_samples(
            data, baseline_vector, train_subjects)
        test_samples, test_labels = generate_samples(
            data, baseline_vector, test_subjects)

        clf = svm.SVC(verbose=1, kernel='linear', C=C, gamma='auto')
        clf.fit(train_samples, train_labels)
        compute_result(clf, train_samples, train_labels)
        compute_result(clf, test_samples, test_labels, True)

    all_samples, all_labels = generate_samples(
            data, baseline_vector, [f"{j}" for j in range(1, 12)])
    clf = svm.SVC(verbose=1, kernel='linear', C=C, gamma='auto')
    clf.fit(all_samples, all_labels)
    print(clf.coef_)
    print(len(all_samples))
    compute_result(clf, all_samples, all_labels)
    joblib.dump(clf, 'svm_model.pkl')




if __name__ == "__main__":
    main()