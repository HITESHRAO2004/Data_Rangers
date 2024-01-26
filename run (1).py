import pickle as pkl
import sys
import pandas as pd
# import sklearn
# from sklearn.preprocessing import LabelEncoder

def load_model():
    with open("model.pkl", "rb") as file:
        model = pkl.load(file)
    return model

def load_label_encoder():
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pkl.load(file)
    return label_encoder

def data_process(test):
    label_encoder = load_label_encoder()

    # test = df.drop(['uuid'], axis=1)
    test['condition'] = label_encoder.transform(test['condition'])

    return test

def predict(test):
    columns = test.columns
    columns =  [col for col in columns if col not in ['HR','uuid']]
    features = columns

    model = load_model()
    y_pred_val = model.predict(test[features])
    return y_pred_val


def save_results(result):
    result[["uuid", "HR"]].to_csv("results.csv",index=False)



def main():
    if len(sys.argv) == 1:
        print("missing commandline input, test data file.")
        print("using default name: test_data.csv")
        test_data = "test_data.csv"
    else:
        test_data = sys.argv[1]

    try:
        data_frame = pd.read_csv(test_data)
    except FileNotFoundError:
        data_frame = pd.read_csv("test_data.csv")
    test = data_process(data_frame)

    results = predict(test)

    test["HR"] = results
    save_results(data_frame)


main()
