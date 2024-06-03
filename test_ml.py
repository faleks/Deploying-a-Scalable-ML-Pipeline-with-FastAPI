import pytest
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, save_model, load_model, inference, compute_model_metrics

@pytest.fixture
def train_model_fixture():
    """
    fixture using train_model.py to run tests
    """
    # load data
    project_path = ""
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    # split data
    train, test = train_test_split(data,random_state=32)

    # process data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,label="salary",training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # train model
    model = train_model(X_train,y_train)

    # save model
    model_path = os.path.join(project_path, "model", "model.pkl")
    save_model(model, model_path)

    # load model
    model = load_model(
        model_path
    ) 

    return model, X_train, y_train, X_test, y_test


def test_model_algorithm(train_model_fixture):
    """
    Test if the ML model uses the expected algorithm
    RandomForestClassifier is expected
    """
    model, _, _, _, _ = train_model_fixture
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(train_model_fixture):
    """
    Test if the computing metrics functions return the expected value
    """
    model, X_train, y_train, X_test, y_test = train_model_fixture
    preds = inference(model, X_test)
    expected_precision = 0.7089285714285715
    expected_recall = 0.6314952279957582
    expected_fbeta = 0.667975322490185
    p, r, fb = compute_model_metrics(y_test, preds)
    assert abs(p - expected_precision) < 0.01
    assert abs(r - expected_recall) < 0.01
    assert abs(fb - expected_fbeta) < 0.01


def test_data_types(train_model_fixture):
    """
    Test if the data types of the datasets are as expected
    """
    model, X_train, y_train, X_test, y_test = train_model_fixture
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
