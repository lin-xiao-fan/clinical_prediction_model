import csv
import collections
import os
from sklearn import svm
import joblib
from sklearn.datasets import load_iris



_DATA_DIR = "clinical_gram"
_NUM_FEATURES = 12

def read_csv( filename ):
    data = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None) #跳過第一行參數名稱
        for row in reader :
            data.append(row)




    return data


def load_data():
    results = collections.defaultdict(dict)
    for file_name in os.listdir(_DATA_DIR):
        parts = file_name.split("_", 4)
        subject = parts[2]
        test = parts[3]
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


def compute_result( data , state   ):
    total = len( data )
    correct = 0 
    rate = 0
    for i in range( 0 , len( data ) ) :    
        if state == "pre" :
            if data[i] == -1 :
                correct += 1
        elif state == "post" :
            if data[i] == 1 :
                correct += 1

    rate += correct / total

    print("Pass rate is %f" % (correct / total))
    return rate



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

    model = joblib.load('svm_model.pkl')
    all = 0.0

    for i in range( 1 , 7 ) :
        for runs in [ "pre" , "post" ] :
        # data = read_csv("test_clinical_%d_%s_post_gram.csv" )
            data = read_csv("test_clinical_%d_%s_gram.csv" % (i, runs)) 
            clinical_predict = model.predict(data)


            print( i , runs )
            print( clinical_predict )
            all += compute_result( clinical_predict, runs  )


    print( "All pass rate : " , all/12 )


if __name__ == "__main__":
    main()