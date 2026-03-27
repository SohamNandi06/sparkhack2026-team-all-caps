import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_partition(hospital_id):
    df = pd.read_csv("data/heart_disease.csv", header=None)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    X = df.iloc[:, 0:13].values
    y = df.iloc[:, 13].values

    y = (y > 0).astype(int)


    ages = X[:, 0]

    if hospital_id == 1:
        idx = ages < 45
    elif hospital_id == 2:
        idx = (ages >= 45) & (ages <= 60)
    elif hospital_id == 3:
        idx = ages > 60
    else:
        raise ValueError("hospital_id must be 1, 2, or 3")

    X = X[idx]
    y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_full_data():
    df = pd.read_csv("data/heart_disease.csv", header=None)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    X = df.iloc[:, 0:13].values
    y = df.iloc[:, 13].values

    y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_test, y_test