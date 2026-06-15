from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data