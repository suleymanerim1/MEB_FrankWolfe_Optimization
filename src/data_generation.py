import numpy as np
import pandas as pd
import scipy.io as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
def daphnet_freezing_data(test_split, scale_data=True, seed=123):
    np.random.seed(seed)

    daphnet_freezing_df = pd.read_csv('datasets/daphnet_freezing.arff', header=None)
    daphnet_freezing_df = daphnet_freezing_df.rename(columns={14: 'y'})

    labels = daphnet_freezing_df['y']
    features = daphnet_freezing_df.drop(columns=['y'])

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

    if scale_data:
        scaler = StandardScaler()  # MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train.T, X_test.T, y_test


def musk_data(test_split, scale_data=True, seed=123):
    np.random.seed(seed)

    musk_df = pd.read_csv('datasets/musk.csv')
    labels = musk_df['Class']
    features = musk_df.drop(columns=['Class'])

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

    if scale_data:
        scaler = StandardScaler()  # MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train.T, X_test.T, y_test


def customer_churn_data(test_split, scale_data=True, seed=123):
    np.random.seed(seed)

    churn_df = pd.read_csv('datasets/CustomerChurn.csv')
    labels = churn_df['Churn']
    features = churn_df.drop(columns=['Churn'])

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

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train.T, X_test.T, y_test


def breast_cancer_data(test_split, scale_data=True, seed=123):
    np.random.seed(seed)

    breastCancer_df = pd.read_csv('datasets/breast_cancer_wisconsin.csv')
    labels = breastCancer_df['diagnosis']
    labels = labels.map({'B': 0, 'M': 1})
    features_to_drop = ['Unnamed: 32', 'id', 'diagnosis']
    features = breastCancer_df.drop(features_to_drop, axis=1)

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

    if scale_data:
        scaler = StandardScaler()  # MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train.T, X_test.T, y_test


def thyroid_data(test_split, scale_data=True, seed=123):
    np.random.seed(seed)

    thyroid_dataset = io.loadmat('datasets/thyroid.mat')

    features = thyroid_dataset['X']  # [:, (0,5)]
    labels = thyroid_dataset['y'].squeeze()
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

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train.T, X_test.T, y_test


# metro_train_data
def metro_train_data(test_split, num_samples=5000, scale_data=True, seed=123):
    np.random.seed(seed)

    metro_train_df = pd.read_csv('datasets/metro_train.csv', nrows=num_samples)
    columns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP']
    metro_train_df = metro_train_df[columns]
    metro_train_df = metro_train_df.rename(columns={"COMP": 'y'})

    # make anomaly as 1 and good as 0 (to make it suitable for algorithms)
    metro_train_df['y'] = metro_train_df['y'].replace({0: 1, 1: 0})

    labels = metro_train_df['y']
    features = metro_train_df.drop('y', axis=1)

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

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train.T, X_test.T, y_test


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
