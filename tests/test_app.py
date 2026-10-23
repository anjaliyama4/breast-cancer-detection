
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_data():
    """Load one valid sample with the expected feature columns."""
    data = load_breast_cancer()
    features = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )
    return features.iloc[[0]]


def test_sample_data_has_expected_columns(sample_data):
    """Check that the sample has all 30 required features."""
    assert sample_data.shape == (1, 30)
    assert list(sample_data.columns) == list(
        load_breast_cancer().feature_names
    )


def test_sample_data_contains_only_finite_numbers(sample_data):
    """Check that measurements are numeric and finite."""
    assert np.isfinite(sample_data.to_numpy()).all()


def test_random_forest_can_predict(sample_data):
    """Check that a model can produce a valid class prediction."""
    data = load_breast_cancer()
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(
    pd.DataFrame(data.data, columns=data.feature_names),
    data.target,
    )

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_random_forest_probability_is_valid(sample_data):
    """Check that predicted probabilities are valid."""
    data = load_breast_cancer()
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(
    pd.DataFrame(data.data, columns=data.feature_names),
    data.target,
    )

    probabilities = model.predict_proba(sample_data)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
