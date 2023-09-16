import numpy as np
import pandas as pd
import scipy.io as io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math


# Data Generation
def create_data(data_config):
    data_creation_method = data_config.get('method')
    m = eval(data_config.get('number_of_samples'))
    n = eval(data_config.get('number_of_variables'))
    test_split = eval(data_config.get('test_split'))
    train_split = 1 - test_split
    train, test_X, test_Y = None, None, None

    if data_creation_method == "random_normal":
        train = generate_random_matrix_normal(0, 1, n, int(m * train_split))
        T_0 = generate_random_matrix_normal(0, 1, n, int(m * test_split * 0.5))
        T_1 = generate_random_matrix_normal(8, 1, n, int(m * test_split * 0.5))
        test_X = np.hstack((T_0, T_1))
        test_Y = [0] * int(m * test_split * 0.5) + [1] * int(m * test_split * 0.5)

    elif data_creation_method == "fermat_spiral":
        train = generate_fermat_spiral(m // 2).T

    elif data_creation_method == "random_uniform":
        train = generate_random_matrix_uniform(0, 0.7, n, int(m * train_split))
        test_X = generate_random_matrix_uniform(0.7, 1, n, int(m * test_split))
        test_Y = [1] * int(m * test_split)

    elif data_creation_method == "daphnet_freezing_data":
        train, test_X, test_Y = daphnet_freezing_data(test_split)

    elif data_creation_method == "metro_train_data":
        train, test_X, test_Y = metro_train_data(test_split)

    elif data_creation_method == "thyroid_data":
        train, test_X, test_Y = thyroid_data(test_split)

    elif data_creation_method == 'breast_cancer_data':
        train, test_X, test_Y = breast_cancer_data(test_split)

    elif data_creation_method == 'musk_data':
        train, test_X, test_Y = musk_data(test_split)

    elif data_creation_method == 'customer_churn_data':
        train, test_X, test_Y = customer_churn_data(test_split)

    return train, (test_X, test_Y)


# daphnet_freezing_data
def daphnet_freezing_data(test_split, seed=123):
    np.random.seed(seed)

    df = pd.read_csv('datasets/daphnet_freezing.arff', header=None)
    df = df.rename(columns={14: 'y'})

    # Create training and testing sets
    train, test = train_test_split(df, test_size=test_split)

    X = train[train.y == 0]

    # Remove generated row names and class column (y)
    X = X.drop(columns=['y'])

    # Normalize data
    scalar = MinMaxScaler()
    train_data = scalar.fit_transform(X).T

    len_good = (test["y"] == 0).sum()
    len_anomaly = (test["y"] == 1).sum()

    # Remove generated row names and class column (y)
    test = test.drop(columns=['y'])

    # normalize data
    test_X = scalar.fit_transform(test).T

    # create a list of zeros and ones for good and anomaly points
    test_Y = [0] * len_good + [1] * len_anomaly

    return train_data, test_X, test_Y


def musk_data(test_split, seed=123):
    np.random.seed(seed)

    medical_df = pd.read_csv('datasets/musk.csv')
    labels = medical_df['Class']
    features = medical_df.drop(columns=['Class'])

    nominal_data = features.loc[labels == 0, :]
    nominal_labels = labels[labels == 0]
    N_nominal = nominal_data.shape[0]

    anomaly_data = features.loc[labels == 1, :]
    anomaly_labels = labels[labels == 1]

    scaler = StandardScaler()
    nominal_data = pd.DataFrame(scaler.fit_transform(nominal_data))
    anomaly_data = pd.DataFrame(scaler.fit_transform(anomaly_data))

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # (1 - test_split) nominal data as training set
    X_train = nominal_data.iloc[randIdx[:N_train]].values

    # test_split nominal data + all novel data as test set
    X_test = np.concatenate((nominal_data.iloc[randIdx[N_train:]], anomaly_data), axis=0)
    y_test = np.concatenate((nominal_labels.iloc[randIdx[N_train:]], anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


def customer_churn_data(test_split, seed=123):
    np.random.seed(seed)

    medical_df = pd.read_csv('datasets/CustomerChurn.csv')
    labels = medical_df['Churn']
    features = medical_df.drop(columns=['Churn'])

    nominal_data = features.loc[labels == 0, :]
    nominal_labels = labels[labels == 0]
    N_nominal = nominal_data.shape[0]

    anomaly_data = features.loc[labels == 1, :]
    anomaly_labels = labels[labels == 1]

    scaler = StandardScaler()
    nominal_data = pd.DataFrame(scaler.fit_transform(nominal_data))
    anomaly_data = pd.DataFrame(scaler.fit_transform(anomaly_data))

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # (1 - test_split) nominal data as training set
    X_train = nominal_data.iloc[randIdx[:N_train]].values

    # test_split nominal data + all novel data as test set
    X_test = np.concatenate((nominal_data.iloc[randIdx[N_train:]], anomaly_data), axis=0)
    y_test = np.concatenate((nominal_labels.iloc[randIdx[N_train:]], anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


def breast_cancer_data(test_split, seed=123):
    np.random.seed(seed)

    medical_df = pd.read_csv('datasets/breast_cancer_wisconsin.csv')
    labels = medical_df['diagnosis']
    labels = labels.map({'B': 0, 'M': 1})
    features_to_drop = ['Unnamed: 32', 'id', 'diagnosis']
    features = medical_df.drop(features_to_drop, axis=1)

    nominal_data = features.loc[labels == 0, :]
    nominal_labels = labels[labels == 0]
    N_nominal = nominal_data.shape[0]

    anomaly_data = features.loc[labels == 1, :]
    anomaly_labels = labels[labels == 1]

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # (1 - test_split) nominal data as training set
    X_train = nominal_data.iloc[randIdx[:N_train]].values

    # test_split nominal data + all novel data as test set
    X_test = np.concatenate((nominal_data.iloc[randIdx[N_train:]], anomaly_data), axis=0)
    y_test = np.concatenate((nominal_labels.iloc[randIdx[N_train:]], anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


def thyroid_data(test_split, seed=123):
    np.random.seed(seed)

    data = io.loadmat('datasets/thyroid.mat')

    features = data['X']  # [:, (0,5)]
    labels = data['y'].squeeze()
    labels = (labels == 0)

    nominal_data = features[labels == 1, :]
    nominal_labels = labels[labels == 1]

    N_nominal = nominal_data.shape[0]

    anomaly_data = features[labels == 0, :]
    anomaly_labels = labels[labels == 0]

    randIdx = np.arange(N_nominal)
    np.random.shuffle(randIdx)

    N_train = int(N_nominal * (1 - test_split))

    # 0.5 nominal data as training set
    X_train = nominal_data[randIdx[:N_train]]

    # 0.5 nominal data + all novel data as test set
    X_test = nominal_data[randIdx[N_train:]]
    y_test = nominal_labels[randIdx[N_train:]]
    X_test = np.concatenate((X_test, anomaly_data), axis=0)
    y_test = np.concatenate((y_test, anomaly_labels), axis=0)

    return X_train.T, X_test.T, y_test


# metro_train_data
def metro_train_data(test_split, seed=123, num_samples=5000):
    np.random.seed(seed)

    df = pd.read_csv('datasets/metro_train.csv')
    columns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP']
    df = df[columns]
    df = df.rename(columns={"COMP": 'y'})

    # make anomaly as 1 and good as 0 (to make it suitable for algorithms)
    df['y'] = df['y'].replace({0: 1, 1: 0})

    df = df.head(num_samples)

    # Create training and testing sets
    train, test = train_test_split(df, test_size=test_split)

    # Normalize data
    scalar = MinMaxScaler()

    X = train[train.y == 0]

    # Remove generated row names and class column (y)
    X = X.drop(columns=['y'])

    train_data = scalar.fit_transform(X).T

    len_good = (test["y"] == 0).sum()
    len_anomaly = (test["y"] == 0).sum()

    # Remove generated row names and class column (y)
    test = test.drop(columns=['y'])

    # normalize data
    test_X = scalar.fit_transform(test).T

    # create a list of zeros and ones for good and anomaly points
    test_Y = [0] * len_good + [1] * len_anomaly

    count_ones = test_Y.count(1)
    print("Number of anomalies (1s) in test_Y:", count_ones)

    return train_data, test_X, test_Y


def generate_fermat_spiral(dot, seed=123):
    np.random.seed(seed)
    data = []
    d = dot * 0.1
    for i in range(dot):
        t = i / d * np.pi
        x = (1 + t) * math.cos(t)
        y = (1 + t) * math.sin(t)
        data.append([x, y])
    narr = np.array(data)
    f_s = np.concatenate((narr, -narr))
    np.random.shuffle(f_s)
    return f_s


def generate_random_matrix_normal(mu, sigma, m, n, seed=123):
    np.random.seed(seed)
    return np.random.normal(loc=mu, scale=sigma, size=(m, n))


def generate_random_matrix_uniform(low, high, m, n, seed=123):
    np.random.seed(seed)
    return np.random.uniform(low, high, size=(m, n))
